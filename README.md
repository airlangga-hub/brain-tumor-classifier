# 🧠 **Brain Tumor Classifier**
This is a web application for *Brain Tumor Classification*.
VISIT the App [here](http://3.106.167.175/)

# 🤖 This project leverages the pre-trained *EfficientNet-B0* model
The model was fine tuned on **brain MRI images**.
Find the dataset [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

# 💻 Tech Stack
* PyTorch
* ONNX
* Streamlit
* Docker
* AWS EC2 Instance

# 🌊 Project Flow
* Import *EfficientNet-B0* model from *torchvision.models*
* Fine tune the model on [brain MRI images](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
* Save the model's parameters as *.pth* file
* Use ONNX to convert the saved parameters into *.onnx* file ➡️ for inference on the cloud
* Create ***inference modul*** for the *.onnx* parameters ➡️ for inference on the cloud
* Containerize using **Docker**
  * command line (build): `docker build -t <image-name> .`
* *Push* the **Docker Image** to *GitHub* and connect it to the repository
  * command line (tag): `docker tag <image-name> ghcr.io/<github-username>/<image-name>:latest`
  * command line (login): `echo <personal-access-token> | docker login ghcr.io -u <github-username> --password-stdin`
  * command line (push): `docker push ghcr.io/<github-username>/<image-name>:latest`
* Create *AWS EC2 Instance*
* Install *Docker* on the *instance*
  * command line (update): `sudo yum update -y`
  * command line (install): `sudo yum install -y docker`
  * command line (start): `sudo service docker start`
* Pull the **Docker Image** from *GitHub* to *AWS*
  * command line (pull): `docker pull ghcr.io/<github-username>/<image-name>:latest`
