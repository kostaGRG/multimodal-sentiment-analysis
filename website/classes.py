from transformers import RobertaModel, ViTModel, AutoModel
import torch
import torch.nn as nn
from transformers import  RobertaTokenizer, AutoTokenizer, ViTImageProcessor


class englishBertClassifier(nn.Module):
    def __init__(self):
        super(englishBertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 100, 3
        dropout = 0.5

        self.bert = RobertaModel.from_pretrained("./models/englishTextModel")
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H), nn.ReLU(), nn.Dropout(dropout), nn.Linear(H, D_out)
        )
        self.classifier.load_state_dict(
            torch.load(
                "./models/englishTextClassifier.pt", map_location=torch.device("cpu")
            )
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


class greekBertClassifier(nn.Module):
    def __init__(self):
        super(greekBertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels

        self.bert = AutoModel.from_pretrained("./models/greekTextModel")
        D_in, H, D_out = 768, 100, 3
        dropout = 0.5
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H), nn.ReLU(), nn.Dropout(dropout), nn.Linear(H, D_out)
        )
        self.classifier.load_state_dict(
            torch.load(
                "./models/greekTextClassifier.pt", map_location=torch.device("cpu")
            )
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


class VisualTransformer(nn.Module):
    def __init__(self):
        super(VisualTransformer, self).__init__()
        D_in, H, D_out = 768, 100, 3
        dropout = 0.2

        self.vit = ViTModel.from_pretrained("./models/imageModel")
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H), nn.ReLU(), nn.Dropout(dropout), nn.Linear(H, D_out)
        )

        self.classifier.load_state_dict(
            torch.load("./models/imageClassifier.pt", map_location=torch.device("cpu"))
        )

    def forward(self, images):
        outputs = self.vit(images)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits


class FourLayersFC(nn.Module):
    def __init__(self, input_size, hidden_size, last_hidden_size, output_size):
        super(FourLayersFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, last_hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(last_hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


def load_models():
    englishBert = englishBertClassifier()
    englishBert.eval()
    greekBert = greekBertClassifier()
    greekBert.eval()
    imageVit = VisualTransformer()
    imageVit.eval()
    multimodalModel = FourLayersFC(input_size=1536,hidden_size=400,last_hidden_size=300,output_size=3)
    multimodalModel.load_state_dict(torch.load('./models/multimodalModel.pt',
                                     map_location=torch.device('cpu')))
    multimodalModel.eval()
    englishTokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    greekTokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
    imageProcessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    return englishBert,greekBert,imageVit,multimodalModel, englishTokenizer, greekTokenizer, imageProcessor