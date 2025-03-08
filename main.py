import streamlit as st
import torchvision.transforms as tf
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
import models
import matplotlib.pyplot as plt


def process_images(image_files, label, transform):
    images = []
    labels = []

    for image_file in image_files:
        image = Image.open(BytesIO(image_file.read())).convert('RGB')
        image_tensor = transform(image)
        images.append(image_tensor)
        labels.append(label)

    return images, labels

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def app():
    st.set_page_config(page_title='ImagePro')
    st.title('ImagePro - Image Classifier')

    if 'device' not in st.session_state:
        st.session_state.device = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'loss_list' not in st.session_state:
        st.session_state.loss_list = None

    with st.expander('Settings'):
        st.subheader('Training')

        devices = {'CPU': 'cpu'}
        if torch.backends.mps.is_available():
            devices['GPU'] = 'mps'
        elif torch.cuda.is_available():
            devices['GPU'] = 'cuda'
        device = devices[st.selectbox('Hardware used for model:', options=devices.keys())]

        learning_type = st.selectbox('Type of learning:', options=['Transfer', 'Novel'])
        learning_rate = float(st.text_input('Rate of learning:', value='0.001'))

        batch_size = int(st.text_input('Number of images per batch:', value='3'))
        epoch_number = int(st.text_input('Number of training epochs:', value='5'))

        shuffle = st.checkbox('Shuffle images for training', value=False)

    image_set_1 = st.file_uploader('Image set 1:', accept_multiple_files=True)
    label_1 = st.text_input('Set 1 label:')

    image_set_2 = st.file_uploader('Image set 2:', accept_multiple_files=True)
    label_2 = st.text_input('Set 2 label:')

    if image_set_1 and label_1 and image_set_2 and label_2:
        with st.spinner('Processing images'):
            label_decode = {0: label_1, 1: label_2}
            label_encode = {v: k for k, v in label_decode.items()}

            transform = tf.Compose([
                tf.Resize((224, 224)),
                tf.RandomHorizontalFlip(),
                tf.RandomRotation(10),
                tf.ToTensor(),
                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            images_1, labels_1 = process_images(image_set_1, label_encode[label_1], transform)
            images_2, labels_2 = process_images(image_set_2, label_encode[label_2], transform)

            all_images = images_1 + images_2
            all_labels = labels_1 + labels_2

        with st.expander('Image set 1'):
            set_len = len(image_set_1)
            if set_len > 1:
                idx = st.slider('Image index:', min_value=0, max_value=set_len - 1)
            else:
                idx = 0
            st.image(image_set_1[idx], caption=f'{label_decode[labels_1[idx]]}')

        with st.expander('Image set 2'):
            set_len = len(image_set_2)
            if set_len > 1:
                idx = st.slider('Image index:', min_value=0, max_value=set_len - 1)
            else:
                idx = 0
            st.image(image_set_2[idx], caption=f'{label_decode[labels_2[idx]]}')

        dataset = ImageDataset(all_images, torch.tensor(all_labels))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        train_model = st.button('Train model')

        if train_model:
            if learning_type == 'Transfer':
                model = models.TransferLearningModel(lr=learning_rate, device=device, pretrained=True)
                loss_list = model.train_model(dataloader, epochs=epoch_number)
            else:
                model = models.TransferLearningModel(lr=learning_rate, device=device, pretrained=False)
                loss_list = model.train_model(dataloader, epochs=epoch_number)

            st.session_state.device = device
            st.session_state.model = model
            st.session_state.loss_list = loss_list

        device = st.session_state.device
        model = st.session_state.model
        loss_list = st.session_state.loss_list

        if model is not None and loss_list is not None:
            fig, ax = plt.subplots()
            ax.plot(range(1, len(loss_list) + 1), loss_list, marker='o', linestyle='-')
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            st.pyplot(fig)
            plt.close(fig)

            test_image = st.file_uploader('Test image:', accept_multiple_files=False)

            if test_image:
                transform = tf.Compose([
                    tf.Resize((224, 224)),
                    tf.ToTensor(),
                    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                image = Image.open(BytesIO(test_image.read())).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)

                prediction, confidence = model.make_prediction(image_tensor)
                label = label_decode[prediction]
                st.image(image, caption=f'{label} ({round(confidence * 100)}% confidence)')


if __name__ == '__main__':
    app()
