import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rnaglib.rnattentional.layers import RGATLayer


class RGATEmbedder(nn.Module):
    """
    This is an exemple RGCN for unsupervised learning, going from one element of "dims" to the other

    It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.
    """

    def __init__(self,
                 dims,
                 num_heads=3,
                 sample_other=0.2,
                 infeatures_dim=0,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 return_loss=True,
                 verbose=False):
        super(RGATEmbedder, self).__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.sample_other = sample_other
        self.use_node_features = (infeatures_dim != 0)
        self.in_dim = 1 if infeatures_dim == 0 else infeatures_dim
        self.conv_output = conv_output
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.verbose = verbose
        self.return_loss = return_loss
        
        self.layers = self.build_model()

        if self.verbose:
            print(self.layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = self.dims[-2:]
        if self.verbose:
            print("short, ", short)
            print("last_hidden, last ", last_hidden, last)

        # input feature is just node degree
        i2h = RGATLayer(in_feat=self.in_dim,
                        out_feat=self.dims[0],
                        num_rels=self.num_rels,
                        num_bases=self.num_bases,
                        num_heads=self.num_heads,
                        sample_other=self.sample_other,
                        activation=F.relu,
                        self_loop=self.self_loop)
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            h2h = RGATLayer(in_feat=dim_in * self.num_heads,
                            out_feat=dim_out,
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            activation=F.relu,
                            self_loop=self.self_loop)
            layers.append(h2h)

        # hidden to output
        if self.conv_output:
            h2o = RGATLayer(in_feat=last_hidden * self.num_heads,
                            out_feat=last,
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            self_loop=self.self_loop,
                            activation=None)
        else:
            h2o = nn.Linear(last_hidden * self.num_heads, last)
        layers.append(h2o)
        return layers

    def deactivate_loss(self):
        for layer in self.layers:
            if isinstance(layer, RGATLayer):
                layer.deactivate_loss()

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g, features,mod = 1):
        iso_loss = 0
        if self.use_node_features:
            if mod == 1:
                h = g.ndata['features'].to(self.current_device)
            else:
                h = features
        else:
            # h = g.in_degrees().view(-1, 1).float().to(self.current_device)
            h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        for i, layer in enumerate(self.layers):
            if not self.conv_output and (i == len(self.layers) - 1):
                h = layer(h)
            else:
                if layer.return_loss:
                    h, loss = layer(g=g, feat=h)
                    iso_loss += loss
                else:
                    #print(features.shape)
                    h = layer(g=g, feat=h)
                    #print(h.shape)
        if self.return_loss:
            return h, iso_loss
        else:
            return h

import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads

        # Define linear transformation layers for query, key, value
        self.q_linear = nn.Linear(self.head_dim, out_features)
        self.k_linear = nn.Linear(self.head_dim, out_features)
        self.v_linear = nn.Linear(self.head_dim, out_features)
        # Dimension transformation
        self.proj_linear = nn.Linear(in_features, self.head_dim * self.num_heads)
    def forward(self, x):
        # Split input matrix x into num_heads parts
        seq_len = x.shape[0]
        # batch_size = x.shape[0]
        x = x.unsqueeze(0)
        batch_size = 1
        #x = self.proj_linear(x)  # (batch_size, seq_len, self.num_heads * self.head_dim)
        x = x.view(batch_size,seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply linear transformation to query, key, value
        q = self.q_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        k = self.k_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        v = self.v_linear(x)  # (batch_size, num_heads, seq_len, head_dim)
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of node features
        out = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(seq_len, -1)  # (batch_size, seq_len, num_heads * head_dim)
        return out.squeeze(0)
import torch
import torch.nn as nn

class MLNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.5):
        super(MLNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.bn3 = nn.BatchNorm1d(hidden_size)
        # self.dropout3 = nn.Dropout(dropout_prob)
        self.out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(torch.relu(self.fc2(x))))
        # x = self.dropout3(self.bn3(torch.relu(self.fc3(x))))
        x = self.sigmoid(self.out(x))
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCrossAttention(nn.Module):
    """

    Args:
        d_seq (int): Input dimension of sequence features (H_l, H_g), also the dimension after projecting structure features (d_seq).
        d_struct (int): Original input dimension of structure features (H_s), which will be projected to d_seq inside the module.
        num_heads (int): Number of heads for multi-head attention, used in both attention stages.
        dropout (float, optional): Dropout rate for attention weights and output. Default: 0.1.
    """
    def __init__(self, d_seq, d_struct, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_seq = d_seq
        self.d_struct = d_struct
        self.num_heads = num_heads
        self.head_dim = d_seq // num_heads
        assert self.head_dim * num_heads == d_seq, "d_seq must be divisible by num_heads"


        self.W1_Q = nn.Linear(d_seq, d_seq)  # Corresponds to W1^Q
        self.W1_K = nn.Linear(d_seq, d_seq)  # Corresponds to W1^K
        self.W1_V = nn.Linear(d_seq, d_seq)  # Corresponds to W1^V
        self.attn_dropout1 = nn.Dropout(dropout)


        self.struct_proj = nn.Linear(d_struct, d_seq)  # Corresponds to W_proj, b_proj

        self.W2_Q = nn.Linear(d_seq, d_seq)  # Corresponds to W2^Q
        self.W2_K = nn.Linear(d_seq, d_seq)  # Corresponds to W2^K
        self.W2_V = nn.Linear(d_seq, d_seq)  # Corresponds to W2^V
        self.attn_dropout2 = nn.Dropout(dropout)


        self.output_proj = nn.Linear(d_seq, d_seq)
        self.output_dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_seq)
        self.layer_norm2 = nn.LayerNorm(d_seq)

    def forward(self, H_l, H_g, H_s):
        """
 

        Args:
            H_l (Tensor): Local sequence features from DCN module. Shape: (batch_size, L, d_seq)
            H_g (Tensor): Global sequence features from RNABert module. Shape: (batch_size, L, d_seq)
            H_s (Tensor): Original structure features from RGAT module. Shape: (batch_size, L, d_struct)

        Returns:
            Z_final (Tensor): Deeply fused feature representation for final binding site prediction.
                               Shape: (batch_size, L, d_seq)
        """
        batch_size, L, _ = H_l.shape


        # H_struct: (batch_size, L, d_struct) -> (batch_size, L, d_seq)
        H_struct = self.struct_proj(H_s)

        Q1 = self.W1_Q(H_l).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)
        K1 = self.W1_K(H_g).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)
        V1 = self.W1_V(H_g).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (scaled dot-product attention)
        scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights1 = F.softmax(scores1, dim=-1)
        attn_weights1 = self.attn_dropout1(attn_weights1)

        # Apply attention weights to Value
        context1 = torch.matmul(attn_weights1, V1)  # (batch_size, num_heads, L, head_dim)
        context1 = context1.transpose(1, 2).contiguous().view(batch_size, L, self.d_seq)

        H_seq = self.layer_norm1(H_l + context1)  # Output: enhanced sequence features H_seq

        Q2 = self.W2_Q(H_seq).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)
        K2 = self.W2_K(H_struct).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)
        V2 = self.W2_V(H_struct).view(batch_size, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights2 = F.softmax(scores2, dim=-1)
        attn_weights2 = self.attn_dropout2(attn_weights2)

        context2 = torch.matmul(attn_weights2, V2)
        context2 = context2.transpose(1, 2).contiguous().view(batch_size, L, self.d_seq)

        # Residual connection and layer normalization
        Z_final = self.layer_norm2(H_seq + context2)

        Z_final = self.output_dropout(F.relu(self.output_proj(Z_final)))

        return Z_final

def bilinear_interpolate(feature_map, sample_points):
    """
    Args:
        feature_map (Tensor): Input feature map, shape (batch, channels, length).
        sample_points (Tensor): Sample point coordinates (float), shape (batch, 1, out_length, kernel_size).

    Returns:
        Tensor: Interpolated feature values, shape (batch, channels, out_length, kernel_size).
    """
    batch, channels, length = feature_map.shape
    # Add channel dimension to sample points for grid sampling
    sample_points = sample_points.unsqueeze(1)  # (batch, 1, out_length, kernel_size) -> (batch, 1, 1, out_length, kernel_size)

    sample_points_normalized = 2.0 * sample_points / (length - 1) - 1.0

    sample_points_normalized = sample_points_normalized.expand(batch, channels, 1, -1, -1)

    sample_points_normalized = sample_points_normalized.permute(0, 1, 2, 4, 3)


    feature_map_2d = feature_map.unsqueeze(2)  # (batch, channels, 1, length)

    batch, channels, out_len, kernel_size = sample_points.shape[0], feature_map.size(1), sample_points.size(2), sample_points.size(3)
    sample_points = sample_points.squeeze(1)  

    # Get integer and fractional parts of sample point coordinates
    x = sample_points
    x0 = x.floor().long()
    x1 = x0 + 1

    # Handle boundaries to prevent index out of bounds
    x0 = torch.clamp(x0, 0, length - 1)
    x1 = torch.clamp(x1, 0, length - 1)

    # Compute weights
    wa = (x1 - x).float()
    wb = (x - x0).float()

    # Collect feature values at indices
    batch_idx = torch.arange(batch, device=feature_map.device).view(batch, 1, 1)
    channel_idx = torch.arange(channels, device=feature_map.device).view(1, channels, 1, 1)

    # Gather features at x0 and x1 positions
    fa = feature_map[batch_idx, channel_idx, x0.unsqueeze(1)]  # (batch, channels, out_len, kernel_size)
    fb = feature_map[batch_idx, channel_idx, x1.unsqueeze(1)]  # (batch, channels, out_len, kernel_size)

    # Linear interpolation
    interpolated = wa.unsqueeze(1) * fa + wb.unsqueeze(1) * fb
    return interpolated


class DeformableConv1D(nn.Module):
    """

    Args:
        in_channels (int): Number of input channels (e.g., nucleotide-level feature dimension d_l = 71).
        out_channels (int): Number of output channels (usually aligned with sequence feature dimension d_seq, e.g., 120).
        kernel_size (int): Size of the convolution kernel K.
        stride (int): Stride, default 1.
        padding (int): Padding, default 0. Can be set to (kernel_size-1)//2 to keep length unchanged.
        dilation (int): Dilation rate, default 1.
        groups (int): Number of groups for grouped convolution, default 1.
        bias (bool): Whether to use bias, default True.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Main convolution weights, used to aggregate features after offset
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Auxiliary convolution layer to learn offsets Δp_k
        # Its output channels equal kernel_size, because each sampling point needs one offset (scalar).
        self.offset_conv = nn.Conv1d(in_channels,
                                     kernel_size,  # Output channels = kernel size, each channel corresponds to one sampling point offset
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=True)  # Offset convolution uses bias

        # Initialize offset convolution weights to zero and bias to zero, so that initial stage is equivalent to standard convolution
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

    def reset_parameters(self):
        """Initialize main convolution weights and bias"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """

        Args:
            x (Tensor): Input feature map, shape (batch_size, in_channels, length).

        Returns:
            output (Tensor): Output feature map, shape (batch_size, out_channels, output_length).
        """
        batch_size, in_channels, length = x.shape
        output_length = (length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        # offset shape: (batch_size, kernel_size, output_length)
        # For each output position l, each convolution kernel sampling point k has an offset.
        offsets = self.offset_conv(x)  # Offsets are continuous floating-point numbers

        # For each output position, the sampling point positions in the corresponding input window.
        center_base = torch.arange(output_length, device=x.device).float() * self.stride
        sampling_offsets_base = torch.arange(self.kernel_size, device=x.device).float() - (self.kernel_size - 1) / 2.0
        sampling_offsets_base = sampling_offsets_base * self.dilation

        # Construct standard positions for all sampling points: p + p_k
        # p: center positions (batch_size, output_length, 1)
        p = center_base.view(1, output_length, 1)
        # p_k: preset offsets (1, 1, kernel_size)
        p_k = sampling_offsets_base.view(1, 1, self.kernel_size)
        standard_locations = p + p_k  # (1, output_length, kernel_size)
        standard_locations = standard_locations + self.padding

        # Apply learned offsets: p + p_k + Δp_k
        # offsets needs to be permuted from (batch, kernel_size, output_length) to (batch, output_length, kernel_size)
        offsets = offsets.permute(0, 2, 1)  # (batch, output_length, kernel_size)
        deformed_locations = standard_locations + offsets  # (batch, output_length, kernel_size)

        # Sample features at deformed locations using bilinear interpolation
        sampled_features = bilinear_interpolate(x, deformed_locations)  # Call the interpolation function defined above

        sampled_features = sampled_features.permute(0, 2, 3, 1)
        output = torch.einsum('blki,oik->blo', sampled_features, self.weight)
        # Adjust output shape: (batch, output_length, out_channels) -> (batch, out_channels, output_length)
        output = output.permute(0, 2, 1)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output

class RGATClassifier(nn.Module):
    """

    It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.
    """

    def __init__(self,
                 rgat_embedder,
                 rbert_embedder = None,
                 rgat_embedder_pre = None,
                 classif_dims=None,
                 num_heads=5,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 verbose=False,
                 return_loss=True,
                 sample_other=0.2):
        super(RGATClassifier, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.conv_output = conv_output
        self.num_heads = num_heads
        self.sample_other = sample_other
        self.return_loss = return_loss
        if rbert_embedder != None:
            self.bert_dim = 120
        else:
            self.bert_dim = 0
        self.feature_dim = 71
        self.rgat_embedder = rgat_embedder
        self.rgat_embedder_pre = rgat_embedder_pre
        self.last_dim_embedder = rgat_embedder.dims[-1] * rgat_embedder.num_heads + self.feature_dim + self.bert_dim + self.feature_dim + 128 * 1
        # noRGCN
        # self.last_dim_embedder =  self.feature_dim + self.bert_dim + self.feature_dim + 128 * 1
        #noBert

        # self.last_dim_embedder = rgat_embedder.dims[-1] * rgat_embedder.num_heads + self.feature_dim  + self.feature_dim + 128 * 1
        #noMid
        # self.last_dim_embedder = rgat_embedder.dims[-1] * rgat_embedder.num_heads + self.bert_dim 
        if self.rgat_embedder_pre!=None:
            self.last_dim_embedder += rgat_embedder_pre.dims[-1] * rgat_embedder_pre.num_heads
        self.classif_dims = classif_dims
        self.rbert_embedder = rbert_embedder	
        self.classif_layers = self.build_model()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.verbose = verbose
        if self.verbose:
            print(self.classif_layers)
            print("Num rels: ", self.num_rels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)

        #################
        self.window = 11
        self.padsize = int(self.window/2)
        self.localpool = nn.MaxPool2d(kernel_size= 11, stride=1)
        self.localpad = nn.ConstantPad2d((self.padsize, self.padsize,self.padsize,self.padsize), value=0)
        
        #att
        att_input =350
        att_output =128
        att_num_head =8
        att_mlpinput = att_output * att_num_head
        att_mlphidden = 32
        self.att = Attention(att_input, att_output, att_num_head)
        self.mlp = MLNet(att_mlpinput, att_mlphidden)
        self.cutoff_len = 440

        kernels = [13, 17, 21]   
        padding1 = (kernels[1]-1)//2
        self.conv2d_1 = torch.nn.Sequential()
        self.conv2d_1.add_module("conv2d_1",torch.nn.Conv2d(1,128,padding= (padding1,0),kernel_size=(kernels[1],self.feature_dim)))
        #self.conv2d.add_module("relu1",nn.BatchNorm2d(128))
        self.conv2d_1.add_module("relu1",torch.nn.ReLU())
        self.conv2d_1.add_module("pool2",torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1))
        #################

        #################
        dcn_out_channels = 120  
        dcn_kernel_size = 3

        self.dcn = nn.Sequential(
            DeformableConv1D(
                in_channels=self.feature_dim,  # Input feature dimension, e.g., 71
                out_channels=dcn_out_channels,  # Output feature dimension, e.g., 120
                kernel_size=dcn_kernel_size,  # Kernel size
                padding=(dcn_kernel_size - 1) // 2,  # Keep length unchanged
                dilation=1,  # Dilation rate
                bias=True
            ),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.dcn_output_dim = dcn_out_channels
        #################

        # padding2 = (kernels[1]-1)//2
        # self.conv2d_2 = torch.nn.Sequential()
        # self.conv2d_2.add_module("conv2d_2",torch.nn.Conv2d(1,128,padding= (padding2,0),kernel_size=(kernels[1],self.feature_dim)))
        # #self.conv2d.add_module("relu1",nn.BatchNorm2d(128))
        # self.conv2d_2.add_module("relu1_2",torch.nn.ReLU())
        # # self.conv2d_2.add_module("pool2_2",torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1))
        # self.conv2d_2_pool = torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1)

        # padding3 = (kernels[2]-1)//2
        # self.conv2d_3 = torch.nn.Sequential()
        # self.conv2d_3.add_module("conv2d_3",torch.nn.Conv2d(1,128,padding= (padding3,0),kernel_size=(kernels[2],self.feature_dim)))
        # #self.conv2d.add_module("relu1",nn.BatchNorm2d(128))
        # self.conv2d_3.add_module("relu1_3",torch.nn.ReLU())
        # self.conv2d_3.add_module("pool2_3",torch.nn.MaxPool2d(kernel_size= (self.cutoff_len,1),stride=1))

        #################
        self.DNN1 = nn.Sequential()
        self.DNN1.add_module("Dense1", torch.nn.Linear(self.last_dim_embedder,192))
        self.DNN1.add_module("Relu1", torch.nn.ReLU())
        self.dropout_layer = nn.Dropout(0.1)
        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("Dense2", torch.nn.Linear(192,96))
        self.DNN2.add_module("Relu2", torch.nn.ReLU())
        self.dropout_layer2 = nn.Dropout(0.1)
        self.outLayer = nn.Sequential(
            torch.nn.Linear(96, 1),
            torch.nn.Sigmoid())
        
        self.fc2d1 = nn.Linear(128, 64)
        self.fc2d2 = nn.Linear(64, 1)
        #################

        #################
        # self.d_seq = 120
        # self.d_struct = 192
        # self.d_l = 71
        #
        # # Instantiate hierarchical cross-attention fusion module
        # self.hca_fusion = HierarchicalCrossAttention(
        #     d_seq=self.d_seq,
        #     d_struct=self.d_struct,
        #     num_heads=8,
        #     dropout=0.1
        # )
        #
        #
        # self.final_predictor = nn.Sequential(
        #     nn.Linear(self.d_seq, 64),  # Input dimension is HCA output d_seq
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        # )
        #################
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for kernel_size in [5, 10, 20]:
            padding = (kernel_size - 1) // 2
            conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=padding)
            bn = nn.BatchNorm1d(32)
            pool = nn.MaxPool1d(kernel_size=2)
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.pool_layers.append(pool)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv11 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=100)
        self.conv12 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=175)
        self.conv13 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=350)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 172, 128)#84 191  64 84
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)
       
    def build_model(self):
        if self.classif_dims is None:
            return self.rgat_embedder

        classif_layers = nn.ModuleList()
        # Just one convolution
        if len(self.classif_dims) == 1:
            if self.conv_output:
                h2o = RGATLayer(in_feat=self.last_dim_embedder,
                                out_feat=self.classif_dims[0],
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                self_loop=self.self_loop,
                                # Old fix for a bug in dgl<0.6
                                # self_loop=self.self_loop and self.classif_dims[0] > 1,
                                activation=None)
            else:
                h2h = nn.Linear(self.last_dim_embedder, self.last_dim_embedder)
                classif_layers.append(h2h)
                h2h2 = nn.Linear(self.last_dim_embedder, self.last_dim_embedder)
                classif_layers.append(h2h2)
                h2o = nn.Linear(self.last_dim_embedder, self.classif_dims[0])
                 
            classif_layers.append(h2o)
            
            return classif_layers

        # The supervised is more than one layer
        else:
            i2h = RGATLayer(in_feat=self.last_dim_embedder,
                            out_feat=self.classif_dims[0],
                            num_rels=self.num_rels,
                            num_bases=self.num_bases,
                            num_heads=self.num_heads,
                            sample_other=self.sample_other,
                            activation=F.relu,
                            self_loop=self.self_loop)
            classif_layers.append(i2h)
            last_hidden, last = self.classif_dims[-2:]
            short = self.classif_dims[:-1]
            for dim_in, dim_out in zip(short, short[1:]):
                h2h = RGATLayer(in_feat=dim_in * self.num_heads,
                                out_feat=dim_out,
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                activation=F.relu,
                                self_loop=self.self_loop)
                classif_layers.append(h2h)

            # hidden to output
            if self.conv_output:
                h2o = RGATLayer(in_feat=last_hidden * self.num_heads,
                                out_feat=last,
                                num_rels=self.num_rels,
                                num_bases=self.num_bases,
                                num_heads=self.num_heads,
                                sample_other=self.sample_other,
                                self_loop=self.self_loop,
                                activation=None)
            else:
                h2o = nn.Linear(last_hidden * self.num_heads, last)
            classif_layers.append(h2o)
            return classif_layers

    def deactivate_loss(self):
        self.return_loss = False
        self.rgat_embedder.deactivate_loss()
        for layer in self.classif_layers:
            if isinstance(layer, RGATLayer):
                layer.deactivate_loss()

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g , features, indxs, seqs, seqlens, len_graphs, chain_idx):
        iso_loss = 0
        g_copy = g.clone().to(self.current_device)
        cnt = 1

        if self.rgat_embedder.return_loss:
            h, loss = self.rgat_embedder(g,features,0)
        else:
            h = self.rgat_embedder(g,features,0)
            loss = 0
        if self.rgat_embedder_pre!=None:
            cnt = 2
            if self.rgat_embedder_pre.return_loss:
                #print(next(self.rgat_embedder_pre.parameters()).device)
            
                h_pre, loss_pre = self.rgat_embedder_pre(g_copy,features,1)
                h = torch.cat((h,h_pre),1)

            else:
                h_pre = self.rgat_embedder_pre(g_copy,features,1)
                h = torch.cat((h,h_pre),1)
                loss = 0
        iso_loss += loss
        # print(features)
        h = torch.cat((h,features),1)
        features = torch.index_select(features, 0, indxs)
        seq_embedding = torch.tensor([]).to(self.current_device).reshape(-1,120)

        h = torch.index_select(h, 0, indxs)

        # noRGCN
        # h = features
        #noMid
        noMid = 1
        # chain_len = torch.tensor([seqlens[x] for x in chain_idx])
        prediction_scores, prediction_scores_ss, seq_encode = self.rbert_embedder(seqs)
        # h = torch.cat((h,features), 1)
        #print(h.shape)
        idx = 0
        for e, seqlen in zip(seq_encode, seqlens):
            if idx in chain_idx:
                seq_embedding = torch.cat((seq_embedding, e[:seqlen].to(self.current_device).reshape( -1, 120)), dim=0)
            idx += 1
        
        pos = 0
        idx = 0
        #print(h.shape)
        single_chain = torch.tensor([]).to(self.current_device).reshape(-1,192*cnt+self.feature_dim*noMid)
        single_features = torch.tensor([]).to(self.current_device).reshape(-1,self.feature_dim)
        for seqlen in seqlens:
            if idx in chain_idx:
                #print(seqlen)
                single_chain = torch.cat((single_chain, h[pos:pos+seqlen].to(self.current_device).reshape( -1, 192*cnt+self.feature_dim*noMid)), dim=0)
                single_features = torch.cat((single_features, features[pos:pos+seqlen].to(self.current_device).reshape( -1, self.feature_dim)), dim=0)
            idx += 1
            pos += seqlen

        #################
        h = torch.cat((single_chain,seq_embedding), 1)
        # h = torch.cat((h,single_features), 1)
        #noBert
        # h = single_chain
        fixed_features = torch.tensor([]).reshape(-1,self.feature_dim).to(self.current_device)
        pos = 0
        idx = 0
        # for graphlen in  len_graphs:
        #for L in  seqlens:
        #print(seqlens)
        #print(chain_idx)
        for idx in chain_idx:
            
            L = seqlens[idx]

            local_h = single_features[pos:pos+L].to(self.current_device)
            #print(local_h.shape)
            for i in range(L,self.cutoff_len):
                add = torch.tensor([[0 for i in range(self.feature_dim)]]).to(self.current_device)
                local_h = torch.cat((local_h,add), 0)
            if  L > self.cutoff_len:
                local_h = local_h[:self.cutoff_len]
            #print(local_h.shape)
            fixed_features=torch.cat((fixed_features, local_h), 0)
            #print(fixed_features.shape)
            pos += L
        #print(len_graphs)
        #print(fixed_features.shape)
        fixed_features = fixed_features.view(-1,1,self.cutoff_len,self.feature_dim)
        g_features = self.conv2d_1(fixed_features)
        # fixed_features_2 = self.conv2d_2(fixed_features)
        # g_features = self.conv2d_2_pool(fixed_features_2)
        # fixed_features_2 = fixed_features_2.squeeze(3)
        # fixed_features_3 = self.conv2d_3(fixed_features).squeeze(3)
        shapes = g_features.shape
        
        g_features = g_features.view(shapes[0],1,shapes[1])
        # print(g_features.shape)
        # print(shapes)
        # fixed_features_1 = fixed_features_1.view(shapes[0],shapes[2],shapes[1])
        # fixed_features_2 = fixed_features_2.view(shapes[0],shapes[2],shapes[1])
        # fixed_features_3 = fixed_features_3.view(shapes[0],shapes[2],shapes[1])
        # L_features = torch.tensor([]).reshape(-1,shapes[1] * 1).to(self.current_device)
        global_features = torch.tensor([]).reshape(-1, g_features.shape[2]).to(self.current_device)
        # fixed_features = torch.cat((fixed_features_2),2)
        # print(fixed_features[1, :seqlens[idx], :].shape)
        cnt = 0
        # for graphlen in  len_graphs:
        for idx in chain_idx:
            
            # L_features = torch.cat((L_features,fixed_features_2[cnt, :seqlens[idx], :]),0)
            for i in range(seqlens[idx]):
                # print(g_features[cnt].shape)
                global_features = torch.cat((global_features,g_features[cnt]),0)
            cnt += 1


        # fixed_features = self.fc2d1(fixed_features)
        # fixed_features = self.relu(fixed_features)
        # fixed_features = self.dropout1(fixed_features)
        # fixed_features = self.sigmoid(self.fc2d2(fixed_features))
        # # fixed_features = self.relu(fixed_features)
        # #print(fixed_features.shape)
        # idx = 0
        # final_features = torch.tensor([]).to(self.current_device)
        # for graphlen in  len_graphs:
        #     local_h = fixed_features[idx][0:graphlen].to(self.current_device)
        #     idx += 1
        #     final_features=torch.cat((final_features, local_h), 0)
            #print(final_features.shape)
        # final_features = self.sigmoid(final_features)
        # return final_features
        
        # # h = h.unsqueeze(0)
        # pos = 0
        # all_features = []

        # for graphlen in  len_graphs:
        #     sub_h = h[pos:pos+graphlen].to(self.current_device)
        #     sub_h = self.att(sub_h)
        #     #print(sub_h.shape)
        #     all_features.append(sub_h)
        # all_features = torch.cat(all_features ,dim = 0)
        # #print(all_features.shape)
        # #result = self.fc2(all_features)
        # #result = self.sigmoid(result)
        # result = self.mlp(all_features)
        # #print(result.shape)
        # return result

        #################
        local_features = torch.tensor([]).reshape(-1,self.feature_dim).to(self.current_device)
        pos = 0
        # for graphlen in  len_graphs:
        for idx in chain_idx:
            L = seqlens[idx]
            local_h = single_features[pos:pos+L].view(1, self.feature_dim, L).to(self.current_device)
           # print(local_h.shape)
            local_h = self.localpool(self.localpad(local_h)).view(L,-1)
            #print(local_h.shape)
            local_features=torch.cat((local_features, local_h), 0)
            pos = pos + L
        # print(local_features.shape)
        # print(global_features.shape)
        # print(h.shape)
        h =  torch.cat((h,local_features,global_features), 1)
        # h =  torch.cat((h,global_features), 1)

        # print(h.shape)
        h = self.DNN1(h)
        h = self.dropout_layer(h)
        h = self.DNN2(h)
        h = self.dropout_layer2(h)
        h = self.outLayer(h)
        #print(h.shape)
        #################

        #################
        # batch_size = len(chain_idx)  # Number of RNA chains in current batch
        # all_seq_lengths = [seqlens[i] for i in chain_idx]  # Actual length of each chain
        # max_len = max(all_seq_lengths)  # Maximum sequence length in the batch
        #
        #
        # device = self.current_device
        # H_s_batch = torch.zeros(batch_size, max_len, self.d_struct, device=device)
        # H_g_batch = torch.zeros(batch_size, max_len, self.d_seq, device=device)
        # features_batch = torch.zeros(batch_size, max_len, self.feature_dim, device=device)
        #
        # # Pad flattened features into batch tensor
        # node_start = 0
        # for i, seq_len in enumerate(all_seq_lengths):
        #
        #     H_s_batch[i, :seq_len, :] = single_chain[node_start:node_start + seq_len, :self.d_struct]
        #
        #     # Global sequence features H_g (from RNABert)
        #     H_g_batch[i, :seq_len, :] = seq_embedding[node_start:node_start + seq_len, :]
        #
        #     # Nucleotide-level features (DCN input)
        #     features_batch[i, :seq_len, :] = single_features[node_start:node_start + seq_len, :]
        #
        #     node_start += seq_len
        #
        #
        # features_batch_transposed = features_batch.permute(0, 2, 1)  # (batch, d_l, max_len)
        # H_l_batch = self.dcn(features_batch_transposed)  # (batch, dcn_output_dim, max_len)
        # H_l_batch = H_l_batch.permute(0, 2, 1)  # (batch, max_len, dcn_output_dim)
        #
        # # Ensure H_l dimension aligns with H_g
        # if self.dcn_output_dim != self.d_seq:
        #     # Add a projection layer
        #     H_l_batch = self.hca_fusion.W1_Q(H_l_batch)  # Project using linear layer from HCA
        #
        # # Deep fusion via HCA module
        # Z_final = self.hca_fusion(H_l_batch, H_g_batch, H_s_batch)  # (batch, max_len, d_seq)
        #
        #
        # predictions_padded = self.final_predictor(Z_final)  # (batch, max_len, 1)
        # predictions_padded = predictions_padded.squeeze(-1)  # (batch, max_len)
        #
        # # Flatten batch predictions to match original label order
        # final_predictions = []
        # for i, seq_len in enumerate(all_seq_lengths):
        #     final_predictions.append(predictions_padded[i, :seq_len])
        #
        # h = torch.cat(final_predictions, dim=0)  # Shape: [total nodes, ]
        #################

        return h

        h = h.unsqueeze(1)

        
        shapes = h.shape
        
        
        #h11 = self.conv11(h)

        #h12 = self.conv12(h)

        #h13 = self.conv13(h) 

        h = self.conv1(h)
        #conv_outs = [h, h11, h12, h13]
        #h = torch.cat(conv_outs, dim=2)
        #print(h.shape)
        #h = self.bn1(h)
        h = self.relu1(h)

        #print(h.shape)
        h = self.pool1(h)
        h = self.conv2(h)
        #print(h.shape)
        #h = self.bn2(h)
        h = self.relu2(h)
        h = self.pool2(h)
        #print(h.shape)
        #h = self.conv2d(h)
        #print(h.shape)
        #shapes = h.data.shape
        h = h.view(shapes[0], -1)
        h = self.fc1(h)
        h = self.relu3(h)
        h = self.dropout1(h)
        h = self.fc2(h)
        h = self.dropout2(h)
        h = self.sigmoid(h)
        print(h.shape)

        return h
        #print(h.shape)
        for i, layer in enumerate(self.classif_layers):
            # if this is the last layer and we want to use a linear layer, the call is different
            if (i == len(self.classif_layers) - 1) and not self.conv_output:
                h = layer(h)
                h = self.sigmoid(h)
            # Convolution layer
            else:
                if self.return_loss:
                    h, loss = layer(g, h)
                    # h = self.relu(h)
                    iso_loss += loss
                else:
                    h = layer(h)
                    h = self.relu(h)

        if self.return_loss:
            return h, iso_loss
        else:
            return h
