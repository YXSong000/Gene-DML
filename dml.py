import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
from models.module import GlobalEncoder, NeighborEncoder, TransformerEncoder
from einops import rearrange
from torch import einsum

def KMeans(x, K=10, Niters=10, verbose=False):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    # start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    # x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)
    x_i = x[:, None, :]  # (Npoints, 1, D)

    for i in range(Niters):
        c_j = c[None, :, :]  # (1, Nclusters, D)
        # c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = cl.view(cl.size(0), 1).expand(-1, D)
        unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)
        # As some clusters don't contain any samples, manually assign count as 1
        labels_count_all = torch.ones([K]).long().to(x.device)
        labels_count_all[unique_labels[:,0]] = labels_count
        c = torch.zeros([K, D], dtype=x.dtype).to(x.device).scatter_add_(0, Ncl, x)
        c = c / labels_count_all.float().unsqueeze(1)

    return cl, c
def embeddings_to_cosine_similarity(E, sigma=1.0):
    '''
    Build a pairwise symmetrical cosine similarity matrix
    diganal is set as zero
    '''

    dot = torch.abs_(torch.mm(E, E.t())) # E[i]E[j]
    norm = torch.norm(E, 2, 1) # ||E[i]||
    x = torch.div(dot, norm) # E[i]E[j]/||E[j]||
    x = torch.div(x, torch.unsqueeze(norm, 0)) # E[i]E[j]/(||E[j]||*||E[i]||)
    x = x.div_(sigma)

    return torch.max(x, x.t()).fill_diagonal_(0)
def kway_normcuts(F, K=2, sigma=1.0):
    # Build similarity matrix W, use cosine similarity
    W = embeddings_to_cosine_similarity(F, sigma=sigma)

    # Build defree matrix
    degree = torch.sum(W, dim=0)

    # Construct normalized Laplacian matrix L
    D_pow = torch.diag(degree.pow(-0.5))
    L = torch.matmul(torch.matmul(D_pow, torch.diag(degree)-W), D_pow)

    _, eigenvector = torch.linalg.eigh(L.float())

    # Normalize eigenvector along each row
    eigvec_norm = torch.matmul(torch.diag(degree.pow(-0.5)), eigenvector)
    eigvec_norm = eigvec_norm/eigvec_norm[0][0]
    eigvec_norm = eigvec_norm.float()
    k_eigvec = eigvec_norm[:,:K]

    return k_eigvec
def spectral_clustering(F, K=10, clusters=10, Niters=10, sigma=1):
    '''
    Input:
        Sample features F, N x D
        K: Number of eigenvectors for K-Means clustering
        clusters: number of K-Means clusters
        Niters: NUmber of iterations for K-Means clustering
    Output:
        cl: cluster label for each sample, N x 1
        c: centroids of each cluster, clusters x D
    '''
    # Get K eigenvectors with K-way normalized cuts 
    k_eigvec = kway_normcuts(F, K=K, sigma=sigma)

    #  Spectral embedding with K eigen vectors
    cl, _ = KMeans(k_eigvec, K=clusters, Niters=Niters, verbose=False)

    # Get unique labels and samples numbers of each cluster
    Ncl = cl.view(cl.size(0), 1).expand(-1, F.size(1))
    unique_labels, labels_count = Ncl.unique(dim=0, return_counts=True)

    # As some clusters don't contain any samples, manually assign count as 1
    labels_count_all = torch.ones([clusters]).long().to(F.device)
    labels_count_all[unique_labels[:,0]] = labels_count

    # Calcualte feature centroids
    c = torch.zeros([clusters, F.size(1)], dtype=F.dtype).to(F.device).scatter_add_(0, Ncl, F)
    c = c / labels_count_all.float().unsqueeze(1)

    return cl, c

    
class ImageEncoder_UNI(nn.Module):
    def __init__(self, data_name='her2st', trainable=False, image_embedding=1024, projection_dim=512, weight="/srv2/yson2999/weights/pytorch_model_uni.bin"):
        super().__init__()
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.model = model
        for p in self.model.parameters(): 
            p.requires_grad = trainable
        self.model.load_state_dict(torch.load(weight))
        
        self.fc_layer = nn.Linear(image_embedding, projection_dim)

        if data_name == 'her2st':

            self.neighbor_encoder = NeighborEncoder(image_embedding, 4, 4, int(image_embedding*4), dropout = 0.3, resolution=(5,5))

            # Global Encoder        
            self.global_encoder = GlobalEncoder(image_embedding, 5, 8, int(image_embedding*4), 0.3, 3)

        elif data_name == 'stnet':

            self.neighbor_encoder = NeighborEncoder(image_embedding, 4, 8, int(image_embedding*4), dropout = 0.3, resolution=(5,5))

            # Global Encoder        
            self.global_encoder = GlobalEncoder(image_embedding, 3, 8, int(image_embedding*2), 0.1, 3)

        else:

            self.neighbor_encoder = NeighborEncoder(image_embedding, 4, 16, int(image_embedding*1), dropout = 0.3, resolution=(5,5))
            self.global_encoder = GlobalEncoder(image_embedding, 2, 16, int(image_embedding*1), 0.1, 3)


    def forward(self, x, x_total, position, neighbor, mask, pid=None, sid=None, test=False):
        # Target tokens
        if test:
            target_token = x
        else:
            target_token = self.model.forward_features(x) # B x 265 x 1536 

        neighbor_token = self.neighbor_encoder(neighbor, mask) # B x 25 x 1536
        
        if pid == None:
            global_token = self.global_encoder(x_total, position.squeeze()).squeeze()  # N x 1536
            if sid != None:
                global_token = global_token[sid]
        else:
            assert pid is not None and sid is not None
            pid = pid.view(-1)
            sid = sid.view(-1)
            global_token = torch.zeros((len(x_total), x_total[0].shape[1])).to(x.device)
            
            pid_unique = pid.unique()
            for pu in pid_unique:
                ind = int(torch.argmax((pid == pu).int()))
                x_g = x_total[ind].unsqueeze(0) # 1 x N x 1536
                pos = position[ind]
                
                emb = self.global_encoder(x_g, pos).squeeze() 
                global_token[pid == pu] = emb[sid[pid == pu]].float()
        target_embedding = self.fc_layer(target_token)
        neighbor_embedding = self.fc_layer(neighbor_token)
        global_embedding = self.fc_layer(global_token)
        return target_token, target_embedding, neighbor_token, neighbor_embedding, global_token, global_embedding # fusion_token
        

class ImageEncoder_UNI_v2(nn.Module):
    def __init__(self, trainable=False, image_embedding=1536, projection_dim=512, weight="/srv2/yson2999/weights/pytorch_model_v2.bin"):
        super().__init__()
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': image_embedding,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        self.model = timm.create_model(
            pretrained=False, **timm_kwargs
        ) 
        for p in self.model.parameters(): 
            p.requires_grad = trainable
        self.model.load_state_dict(torch.load(weight))
        
        self.fc_layer = nn.Linear(image_embedding, projection_dim)

        self.neighbor_encoder = NeighborEncoder(image_embedding, 4, 4, int(image_embedding*4), dropout = 0.3, resolution=(5,5))

        self.global_encoder = GlobalEncoder(image_embedding, 5, 8, int(image_embedding*4), 0.3, 3)

    def forward(self, x, x_total, position, neighbor, mask, pid=None, sid=None, test=False):
        # Target tokens
        if test:
            target_token = x
        else:
            target_token = self.model.forward_features(x) # B x 265 x 1536 

        # Neighbor tokens
        neighbor_token = self.neighbor_encoder(neighbor, mask) # B x 25 x 1536
        
        # Global tokens (128, 1536)
        if pid == None:
            global_token = self.global_encoder(x_total, position.squeeze()).squeeze()  # N x 1536
            if sid != None:
                global_token = global_token[sid]
        else:
            assert pid is not None and sid is not None
            pid = pid.view(-1)
            sid = sid.view(-1)
            global_token = torch.zeros((len(x_total), x_total[0].shape[1])).to(x.device)
            
            pid_unique = pid.unique()
            for pu in pid_unique:
                ind = int(torch.argmax((pid == pu).int()))
                x_g = x_total[ind].unsqueeze(0) # 1 x N x 1536
                pos = position[ind]
                
                emb = self.global_encoder(x_g, pos).squeeze() 
                global_token[pid == pu] = emb[sid[pid == pu]].float()
        target_embedding = self.fc_layer(target_token)
        neighbor_embedding = self.fc_layer(neighbor_token)
        global_embedding = self.fc_layer(global_token)
        return target_token, target_embedding, neighbor_token, neighbor_embedding, global_token, global_embedding # fusion_token


    
    
class ImageEncoder_resnet101(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet101', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_resnet152(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet152', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class ImageEncoder_ViT(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_base_patch32_224", pretrained=False, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

class ImageEncoder_CLIP(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_base_patch32_224_clip_laion2b", pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class ImageEncoder_ViT_L(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="vit_large_patch32_224_in21k", pretrained=False, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ImageEncoder_resnet50(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class PreNorm(nn.Module):
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if 'x_kv' in kwargs.keys():
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])
         
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MMAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            residual=True,
            residual_conv_kernel=33,
            eps=1e-8,
            dropout=0.,
            attn_mode='full'
    ):
        """

        Args:
            dim:
            dim_head:
            heads:
            residual:
            residual_conv_kernel:
            eps:
            dropout:
            attn_mode: ['full', 'partial', 'cross']
                'full': All pairs between P and H
                'partial': P->P, H->P, P->H
                'cross': P->H, H->P
                'self': P->P, H->H
        """
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.residual = residual
        self.attn_mode = attn_mode

        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def set_attn_mode(self, attn_mode):
        self.attn_mode = attn_mode

    def forward(self, x, mask=None, return_attn=False, target_num=265, neighbor_num=25, global_num=1):
        b, n, _, h, eps = *x.shape, self.heads, self.eps
        self.num_image = target_num + neighbor_num + global_num

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        q_image = q[:, :, :self.num_image, :] # bs x head x num_image x dim
        k_image = k[:, :, :self.num_image, :]

        q_gene = q[:, :, self.num_image:, :] # bs x head x num_gene x dim
        k_gene = k[:, :, self.num_image:, :]



        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_image = einsum(einops_eq, q_image, k_gene)
        attn_gene = einsum(einops_eq, q_gene, k_gene)
        cross_attn_gene = einsum(einops_eq, q_gene, k_image)
        attn_image = einsum(einops_eq, q_image, k_image)

        # softmax
        pre_softmax_cross_attn_image = cross_attn_image
        if self.attn_mode == 'full': # H->P, P->H, P->P, H->H
            cross_attn_image = cross_attn_image.softmax(dim=-1)
            attn_image_pathways = torch.cat((cross_attn_image, attn_image), dim=-1).softmax(dim=-1)
            attn_gene_histology = torch.cat((attn_gene, cross_attn_gene), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways = attn_gene_histology @ v
            out_histology = attn_image_pathways @ v
        elif self.attn_mode == 'cross': # P->H, H->P
            cross_attn_image = cross_attn_image.softmax(dim=-1)
            cross_attn_gene = cross_attn_gene.softmax(dim=-1)

            # compute output
            out_pathways = cross_attn_gene @ v[:, :, self.num_pathways:]
            out_histology = cross_attn_image @ v[:, :, :self.num_pathways]
        elif self.attn_mode == 'self': # P->P, H->H (Late fusion)
            attn_image = attn_image.softmax(dim=-1)
            attn_gene = attn_gene.softmax(dim=-1)

            out_pathways = attn_gene @ v[:, :, :self.num_pathways]
            out_histology = attn_image @ v[:, :, self.num_pathways:]
        elif self.attn_mode == 'partial': # H->P, P->H, P->P (SURVPATH)
            cross_attn_image = cross_attn_image.softmax(dim=-1)
            attn_gene_histology = torch.cat((attn_gene, cross_attn_gene), dim=-1).softmax(dim=-1)

            # compute output
            out_pathways = attn_gene_histology @ v
            out_histology = cross_attn_image @ v[:, :, :self.num_pathways]
        elif self.attn_mode == 'mcat': # P->P, P->H
            cross_attn_gene = cross_attn_gene.softmax(dim=-1)

            out_pathways = q_gene
            out_histology = cross_attn_gene @ v[:, :, self.num_pathways:]
        else:
            raise NotImplementedError(f"Not implemented for {self.mode}")

        out = torch.cat((out_pathways, out_histology), dim=2)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)

        if return_attn:
            # return three matrices
            return out, attn_gene.squeeze().detach().cpu(), cross_attn_gene.squeeze().detach().cpu(), pre_softmax_cross_attn_image.squeeze().detach().cpu()

        return out



class GeneEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim=250, #512
        projection_dim=512, #512 original
        dropout=0.3
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"reduction '{reduction}' not supported")

def contrastive_loss(emb1, emb2, temperature, reduction='mean'):
    # Normalize embeddings
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)
    logits = torch.matmul(emb1, emb2.T) / temperature
    labels = torch.arange(emb1.size(0)).to(emb1.device)
    loss_i = F.cross_entropy(logits, labels, reduction=reduction)
    loss_j = F.cross_entropy(logits.T, labels, reduction=reduction)
    return (loss_i + loss_j) / 2


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

class GeneDML(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        lambda_cld=0.8,
        image_embedding=1024, #512
        spot_embedding=250,
        projection_dim=512,
        contrastive_weight=1.0,
        prediction_weight=1.0,
        training_mode='both',  # 'contrastive', 'prediction', 'both'
        size='base',
        cld_t=0.07,
        data_name='her2st',
        weight_dir='/srv2/yson2999/weights',
        test=False
    ):
        super().__init__()
        self.cld_t = cld_t
        if size == 'base':
            image_embedding = 1024
            self.image_encoder = ImageEncoder_UNI(data_name=data_name, image_embedding=image_embedding, projection_dim=projection_dim, weight=f"{weight_dir}/pytorch_model_uni.bin")
        else:
            image_embedding = 1536
            self.image_encoder = ImageEncoder_UNI_v2(image_embedding=image_embedding, projection_dim=projection_dim, weight="/srv2/yson2999/weights/pytorch_model_v2.bin")
        self.gene_encoder = nn.Sequential(GeneEncoder(embedding_dim=spot_embedding, projection_dim=projection_dim), FeedForward(emb_dim=projection_dim, hidden_dim=projection_dim*2, dropout=0.1))
        self.temperature = temperature
        self.lambda_cld = lambda_cld
        self.contrastive_weight = contrastive_weight
        self.prediction_weight = prediction_weight
        self.training_mode = training_mode
        self.trans_encoder = nn.Sequential(TransformerEncoder(emb_dim=projection_dim, depth=1, heads=1, mlp_dim=projection_dim*2, dropout=0.1), FeedForward(emb_dim=projection_dim, hidden_dim=projection_dim*2, dropout=0.1))
        self.identity = nn.Identity()

        self.target_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, spot_embedding)
        )
        self.neighbor_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, spot_embedding)
        )
        self.global_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, spot_embedding)
        )
        
        # Fusion predictor (ensemble of all image features)
        self.fusion_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 3, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, spot_embedding)
        )
        # ablation study for rebuttal
        self.feature_cluster = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            Normalize(2)
        )

    def forward(self, patch, exp, x_total, position, neighbor, mask, pid=None, sid=None, test=False, use_kmeans=True, k=10):
        batch_size = patch.shape[0]
        # WSI
        target_token, target_embedding, neighbor_token, neighbor_embedding, global_token, global_embedding = self.image_encoder(patch, x_total, position, neighbor, mask, pid, sid)
        
        # Gene
        if not test:
            gene_token = self.gene_encoder(exp) # B x projection_dim
        
        # concat target with gene
        image_tokens = torch.cat([target_embedding, neighbor_embedding, global_embedding.unsqueeze(1)], dim=1) # B x 291 x 512
        image_tokens = self.identity(image_tokens)
        image_tokens = self.trans_encoder(image_tokens)


        target_tokens = torch.mean(image_tokens[:, :target_embedding.shape[1], :], dim=1)
        neighbor_tokens = torch.mean(image_tokens[:, target_embedding.shape[1]:target_embedding.shape[1]+neighbor_embedding.shape[1], :], dim=1)
        global_tokens = image_tokens[:, -1, :]

            
        


        # Calculating the Loss
        target_tokens = F.normalize(target_tokens, dim=-1)
        neighbor_tokens = F.normalize(neighbor_tokens, dim=-1)
        global_tokens = F.normalize(global_tokens, dim=-1)
        if not test:
            gene_token = F.normalize(gene_token, dim=-1)
            
            # contrastive_loss = contrastive_loss(gene_token, target_tokens, self.temperature)
            
            logits = (gene_token @ target_tokens.T) / self.temperature
            target_similarity = target_tokens @ target_tokens.T
            spots_similarity = gene_token @ gene_token.T
            targets = F.softmax(
                ((target_similarity + spots_similarity) / 2) / self.temperature, dim=-1
            )
            spots_loss = cross_entropy(logits, targets, reduction='none')
            targets_loss = cross_entropy(logits.T, targets.T, reduction='none')
            target_loss =  (targets_loss + spots_loss) / 2.0 # shape: (batch_size)


            logits = (gene_token @ neighbor_tokens.T) / self.temperature
            target_similarity = neighbor_tokens @ neighbor_tokens.T
            spots_similarity = gene_token @ gene_token.T
            targets = F.softmax(
                ((target_similarity + spots_similarity) / 2) / self.temperature, dim=-1
            )
            spots_loss = cross_entropy(logits, targets, reduction='none')
            neighbor_loss = cross_entropy(logits.T, targets.T, reduction='none')
            neighbor_loss =  (neighbor_loss + spots_loss) / 2.0 # shape: (batch_size)


            logits = (gene_token @ global_tokens.T) / self.temperature
            target_similarity = global_tokens @ global_tokens.T
            spots_similarity = gene_token @ gene_token.T
            targets = F.softmax(
                ((target_similarity + spots_similarity) / 2) / self.temperature, dim=-1
            )
            spots_loss = cross_entropy(logits, targets, reduction='none')
            global_loss = cross_entropy(logits.T, targets.T, reduction='none')
            global_loss =  (global_loss + spots_loss) / 2.0 # shape: (batch_size)

            contrastive_loss = (target_loss + neighbor_loss + global_loss) / 3.0

        target_pred = self.target_predictor(target_tokens)
        neighbor_pred = self.neighbor_predictor(neighbor_tokens)
        global_pred = self.global_predictor(global_tokens)
        fusion_pred = self.fusion_predictor(torch.mean(image_tokens, dim=1))
        if not test:
            target_pred_loss = F.mse_loss(target_pred, exp)
            neighbor_pred_loss = F.mse_loss(neighbor_pred, exp)
            global_pred_loss = F.mse_loss(global_pred, exp)
            fusion_pred_loss = F.mse_loss(fusion_pred, exp)
                
            prediction_loss = fusion_pred_loss
            loss = contrastive_loss.mean() + prediction_loss.mean()

        if test:
            return {
                # 'contrastive_loss': contrastive_loss.mean(),
                # 'prediction_loss': prediction_loss,
                # 'target_loss': target_loss,
                # 'neighbor_loss': neighbor_loss,
                # 'global_loss': global_loss,
                # 'target_pred_loss': target_pred_loss,
                # 'neighbor_pred_loss': neighbor_pred_loss,
                # 'global_pred_loss': global_pred_loss,
                # 'fusion_pred_loss': fusion_pred_loss,
                'predictions': {
                    'target': target_pred,
                    'neighbor': neighbor_pred,
                    'global': global_pred,
                    'fusion': fusion_pred
                }
            }
        else:
            criterion_cld = torch.nn.CrossEntropyLoss()
            image_clustering_features = self.feature_cluster(torch.mean(image_tokens, dim=1))
            gene_clustering_features = self.feature_cluster(gene_token)
            
            if use_kmeans:
                print('Using KMeans')
                cluster_label1, centroids1 = KMeans(image_clustering_features, K=k)
                cluster_label2, centroids2 = KMeans(gene_clustering_features, K=k)
            else:
                print('Using Spectral Clustering')
                cluster_label1, centroids1 = spectral_clustering(image_clustering_features, K=k, clusters=k)
                cluster_label2, centroids2 = spectral_clustering(gene_clustering_features, K=k, clusters=k)

            images_group_similarity = torch.mean(image_tokens, dim=1) @ centroids2.T
            CLD_loss = criterion_cld(images_group_similarity.div_(self.cld_t), cluster_label2)

            exp_group_similarity = gene_token @ centroids1.T
            CLD_loss = (CLD_loss + criterion_cld(exp_group_similarity.div_(self.cld_t), cluster_label1))/2

            # get cluster label prediction accuracy
            _, cluster_pred = torch.topk(images_group_similarity, 1)
            cluster_pred = cluster_pred.t()
            correct = cluster_pred.eq(cluster_label2.view(1, -1).expand_as(cluster_pred))
            correct_all = correct[0].view(-1).float().sum(0).mul_(100.0/batch_size)
            loss = loss + self.lambda_cld * CLD_loss.mean()
        
            return loss, {
                'contrastive_loss': contrastive_loss.mean(),
                'prediction_loss': prediction_loss,
                'CLD_loss': CLD_loss,
                'target_pred_loss': target_pred_loss,
                'neighbor_pred_loss': neighbor_pred_loss,
                'global_pred_loss': global_pred_loss,
                'fusion_pred_loss': fusion_pred_loss,
                'predictions': {
                    'target': target_pred,
                    'neighbor': neighbor_pred,
                    'global': global_pred,
                    'fusion': fusion_pred
                }
            }   
           