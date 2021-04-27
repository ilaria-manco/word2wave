class MinMaxScaler():
    """
    Transforms each channel to the range [0, 1].
    """    
    def __call__(self, tensor):
        for ch in tensor:
            scale = 1.0 / (ch.max(dim=0)[0] - ch.min(dim=0)[0])        
            ch.mul_(scale).sub_(ch.min(dim=0)[0])        
        return tensor

def compute_audio_embedding_with_scaler(spec, audio_model):
    scaler = pickle.load(open('/content/coala/scaler_top_1000.pkl', 'rb'))
    x = spec.detach().cpu().numpy()
    x = scaler.transform(x)
    # x = torch.unsqueeze(torch.unsqueeze(torch.tensor(x), 0), 0).float().cuda()
    embedding, embedding_d = audio_model(torch.from_numpy(x).cuda().unsqueeze(0).unsqueeze(0))
    return embedding_d

def embeddings_to_cosine_similarity_matrix(z):
    """Converts a a tensor of n embeddings to an (n, n) tensor of similarities.
    """
    cosine_similarity = torch.matmul(z, z.t())
    embedding_norms = torch.norm(z, p=2, dim=1)
    embedding_norms_mat = embedding_norms.unsqueeze(0)*embedding_norms.unsqueeze(1)
    cosine_similarity = cosine_similarity / (embedding_norms_mat)
    return cosine_similarity

def contrastive_loss(text, tag_model, audio, audio_model, t=1):
    z_tag = compute_tag_embedding(text, tag_model).unsqueeze(0)
    z_audio = compute_audio_embedding(audio, audio_model)

    z = torch.cat((z_audio, z_tag), dim=0)
    s = embeddings_to_cosine_similarity_matrix(z)
    N = int(s.shape[0]/2)
    s = torch.exp(s/t)
    try:
        s = s * (1 - torch.eye(len(s), len(s)).cuda())
        # s[range(len(s)), range(len(s))] = torch.zeros((len(s),)).cuda()
    except AssertionError:
        s = s * (1 - torch.eye(len(s), len(s)))
    denom = s.sum(dim=-1)
    num = torch.cat((s[:N,N:].diag(), s[N:,:N].diag()), dim=0)
    return torch.log((num / denom) + 1e-5).neg().mean()

def sample_noise(size, latent_dim=100):
    noise = torch.FloatTensor(size, latent_dim).to("cuda")
    noise.data.normal_()
    # noise.data.uniform_()
    return noise
