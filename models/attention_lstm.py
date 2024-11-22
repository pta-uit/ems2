import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended_values = hidden_states * attention_weights
        context_vector = torch.sum(attended_values, dim=1)
        return context_vector, attention_weights

class AttentionLSTM(nn.Module):
    def __init__(self, args, data):
        super(AttentionLSTM, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidden_dim = args.hidRNN
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.m,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention Layer
        self.attention = AttentionLayer(self.hidden_dim)
        
        # Output Layer
        self.fc = nn.Linear(self.hidden_dim, args.horizon)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Output activation
        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        elif args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out)
        
        # Apply dropout
        context_vector = self.dropout_layer(context_vector)
        
        # Final prediction
        out = self.fc(context_vector)
        
        # Apply output activation if specified
        if self.output:
            out = self.output(out)
            
        return out