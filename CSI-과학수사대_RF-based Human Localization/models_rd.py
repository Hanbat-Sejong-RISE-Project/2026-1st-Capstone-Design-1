import torch
import torch.nn as nn


MAX_PEOPLE = 3
NUM_COUNT_CLASSES = 4


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class MultiTaskResNet(nn.Module):
    """Shared ResNet backbone + count classification head + people-specific location heads."""

    def __init__(self, in_channels, base_channels=32, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, 2, 1, dropout)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, 2, dropout)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, 2, dropout)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, 2, dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.feature_dim = base_channels * 8
        hidden_dim = base_channels * 4
        self.count_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_COUNT_CLASSES),
        )
        self.location_heads = nn.ModuleDict(
            {
                str(people): nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, people * 2),
                )
                for people in range(1, MAX_PEOPLE + 1)
            }
        )
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = [BasicBlock(in_channels, out_channels, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def extract_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.flatten(x)

    def forward(self, x):
        features = self.extract_features(x)
        return {"features": features, "count_logits": self.count_head(features)}

    def predict_location_head(self, features, people):
        people = int(people)
        if people < 1 or people > MAX_PEOPLE:
            raise ValueError(f"위치 head는 people=1,2,3만 지원합니다: {people}")
        return self.location_heads[str(people)](features)

    def route_locations(self, features, counts, fill_value=float("nan")):
        routed = torch.full(
            (features.shape[0], MAX_PEOPLE * 2),
            fill_value=fill_value,
            device=features.device,
            dtype=features.dtype,
        )
        for people in range(1, MAX_PEOPLE + 1):
            mask = counts == people
            if mask.any():
                routed[mask, : people * 2] = self.predict_location_head(features[mask], people)
        return routed

    @torch.no_grad()
    def infer(self, x):
        outputs = self.forward(x)
        count_logits = outputs["count_logits"]
        count_probs = torch.softmax(count_logits, dim=1)
        count_preds = count_logits.argmax(dim=1)
        return {
            "count_logits": count_logits,
            "count_probs": count_probs,
            "count_preds": count_preds,
            "location_preds": self.route_locations(outputs["features"], count_preds),
        }


def build_multitask_resnet(in_channels, base_channels=32, dropout=0.1):
    return MultiTaskResNet(in_channels=in_channels, base_channels=base_channels, dropout=dropout)
