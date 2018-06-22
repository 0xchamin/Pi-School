class PreNet(gluon.Block):
    def __init__(self, **kwargs):
        super(PreNet, self).__init__(**kwargs)
        
        with self.name_scope():
            self.fc1 = nn.Dense(256, activation='relu', flatten=False)
            self.dp1 = nn.Dropout(rate=0.5)
            self.fc2 = nn.Dense(128, activation='relu', flatten=False)
            self.dp2 = nn.Dropout(rate=0.5)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        return x