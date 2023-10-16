# ChurnPredict Pro - Real-time Customer Churn Prediction Web Application 📈📊

ChurnPredict Pro is a powerful web application built on top of a Random Forest Classifier model, designed to predict customer churn. It provides businesses with real-time insights into customer retention and helps optimize customer management strategies 💼💰🤖

## Table of Contents 📚

- [Introduction](#introduction) 📝
- [Features](#features) ✨
- [Demo](#demo) 🚀
- [Getting Started](#getting-started) 🏁
  - [Installation](#installation) 🛠️
  - [Running the App](#running-the-app) 🏃
- [App Structure](#app-structure) 🧱
- [Usage](#usage) 📊
  - [Making Predictions](#making-predictions) 📈
- [Technologies Used](#technologies-used) 💻🔬
- [Contributing](#contributing) 🤝🙌
- [License](#license) 📜

## Introduction 🚀

ChurnPredict Pro uses a state-of-the-art Random Forest Classifier model to predict customer churn. It offers a user-friendly interface for inputting customer data and receiving instant churn predictions.

## Features ✨

- Real-time customer churn predictions.
- Interactive user interface.
- Easy-to-use design.

## Demo 🚀

- ### Pictures 📸
  | ![app header](https://github.com/snyamson/P4-ChurnPredict-Pro/assets/58486437/75cac65c-9184-4660-8da6-95d4c81f7cc2) | ![more cus info](https://github.com/snyamson/P4-ChurnPredict-Pro/assets/58486437/26daa7ff-91dc-4a7a-af61-8ca376e2bb00) |
  | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  |![submit and pred](https://github.com/snyamson/P4-ChurnPredict-Pro/assets/58486437/275209cd-2bb4-4201-82bd-115df186a81d)        | ![pred](https://github.com/snyamson/P4-ChurnPredict-Pro/assets/58486437/b9193492-d28a-47df-9979-6f6d9ca8975f)                 |

- ### Article Link 🌐
  [Read Article](https://www.linkedin.com/pulse/churnpredict-pro-customer-churn-prediction-app-gradio-solomon-wolaf)

## Getting Started 🏁

Follow these instructions to get the app up and running on your local machine.

### Installation 🛠️

1. Clone the repository:

   ```bash
   git clone https://github.com/snyamson/P4-ChurnPredict-Pro.git
   cd P4-ChurnPredict-Pro
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the App 🏃

Run the Gradio app using the following command:

```bash
python src/app/app.py
```

Access the app through your web browser at `http://localhost:7860`.

## App Structure 🧱

- `src`: The main application directory.
- `app/`: Directory containing the main application script `app.py`.
- `model/`: Directory for storing the pre-trained Random Forest Classifier model and preprocessing tools.
- `notebook/`: Directory containing data preprocessing details and model training.

## Usage 📊

### Making Predictions 📈

1. Fill in the customer data in the required fields.
2. Click the "Submit" button to receive a real-time churn prediction.

## Technologies Used 💻🔬

- Gradio: Python library for building interactive interfaces.
- Pandas: Data manipulation and analysis library.
- Scikit-Learn: Machine learning library.

## Contributing 🤝🙌

Contributions to the ChurnPredict Pro project are welcome. Please follow these guidelines for contributing:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b feature-name`
3. Make your changes and commit them with clear, concise commit messages.
4. Push your changes to your fork.
5. Create a pull request against the main repository.

## License📜

This project is licensed under the [MIT License](LICENSE).

## Author✍️

Solomon Nyamson

Connect with me on LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/solomon-nyamson/)

---

Feel free to star ⭐ this repository if you find it helpful!
