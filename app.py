
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Neural Diagnostics | MLP Lab 06",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0B1120;
    color: #E5EAF2;
}

[data-testid="stSidebar"] {
    background-color: #101A2B;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #DCE5F0;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #F1F5F9 !important;
    font-weight: 700 !important;
}

p, li, label {
    color: #C1CCDA;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #273B53;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #92A7BC !important;
}

[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
}

.stButton > button {
    background: #16A6A0;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.65rem 1.2rem;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #128780;
    color: white;
    border: none;
}

div[data-testid="stTabs"] button {
    color: #B7C6D8;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #43D5C6;
}

hr {
    border-color: #29384C;
}

[data-testid="stExpander"] {
    border: 1px solid #293B52;
    border-radius: 10px;
    background: #111D30;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293B52;
    border-radius: 10px;
}

.hero {
    background: linear-gradient(115deg, #142A40, #173C4B);
    border: 1px solid #31546A;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.eyebrow {
    color: #50D4C5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: #F5F8FC;
    font-family: 'Manrope', sans-serif;
    font-size: 36px;
    font-weight: 800;
    margin-top: 8px;
}

.hero-subtitle {
    color: #B6C9D8;
    font-size: 15px;
}

.info-card {
    background: #142238;
    border: 1px solid #2A4059;
    padding: 20px;
    border-radius: 12px;
    min-height: 120px;
}

.info-title {
    color: #55D6C7;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

.info-body {
    color: #D5DFEA;
    font-size: 14px;
    line-height: 1.7;
}

.section-label {
    color: #55D6C7;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.small-muted {
    color: #8EA2B8;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. DATASET LOADING
# ============================================================

@st.cache_data
def load_dataset():

    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    return X, y, dataset.target_names, dataset.feature_names


X, y, target_names, feature_names = load_dataset()


# ============================================================
# 4. SESSION STATE
# ============================================================

if "trained" not in st.session_state:
    st.session_state.trained = False

if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = None


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def build_model(input_neurons, hidden_layers, activation, dropout):

    model = tf.keras.Sequential(name="MLP_Classifier")

    model.add(
        tf.keras.layers.Input(shape=(input_neurons,))
    )

    for i, neurons in enumerate(hidden_layers):

        model.add(
            tf.keras.layers.Dense(
                neurons,
                activation=activation,
                name=f"hidden_layer_{i+1}"
            )
        )

        if dropout > 0:
            model.add(
                tf.keras.layers.Dropout(dropout)
            )

    model.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="output_layer"
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def calculate_parameters(input_neurons, hidden_layers):

    rows = []
    previous = input_neurons

    for i, neurons in enumerate(hidden_layers + [1]):

        weights = previous * neurons
        biases = neurons
        total = weights + biases

        if i == len(hidden_layers):
            name = "Output Layer"
        else:
            name = f"Hidden Layer {i+1}"

        rows.append({
            "Layer": name,
            "Input Neurons": previous,
            "Output Neurons": neurons,
            "Weights": weights,
            "Biases": biases,
            "Total Parameters": total
        })

        previous = neurons

    return pd.DataFrame(rows)


def create_architecture(input_neurons, hidden_layers):

    layer_sizes = [input_neurons] + hidden_layers + [1]

    fig, ax = plt.subplots(figsize=(12, 5))

    fig.patch.set_facecolor("#142238")
    ax.set_facecolor("#142238")

    colors = ["#287F9B"] + ["#16A6A0"] * len(hidden_layers) + ["#D6A34B"]

    x_positions = np.linspace(0, 1, len(layer_sizes))

    for i, size in enumerate(layer_sizes):

        display_count = min(size, 8)

        y_positions = np.linspace(
            0.2, 0.8, display_count
        )

        for ypos in y_positions:

            ax.scatter(
                x_positions[i],
                ypos,
                s=220,
                color=colors[i],
                edgecolors="#D8F1EF",
                linewidths=0.7,
                zorder=3
            )

        if size > display_count:
            ax.text(
                x_positions[i],
                0.08,
                f"+{size-display_count} more",
                ha="center",
                color="#AFC1D3",
                fontsize=8
            )

        if i < len(layer_sizes) - 1:

            next_display = min(layer_sizes[i+1], 8)
            next_y = np.linspace(0.2, 0.8, next_display)

            for ypos in y_positions:
                for next_ypos in next_y:
                    ax.plot(
                        [x_positions[i], x_positions[i+1]],
                        [ypos, next_ypos],
                        color="#54728B",
                        alpha=0.35,
                        linewidth=0.6,
                        zorder=1
                    )

        ax.text(
            x_positions[i],
            0.95,
            f"{size} neurons",
            ha="center",
            color="#E5EAF2",
            fontsize=9,
            fontweight="bold"
        )

        if i == 0:
            label = "INPUT"
        elif i == len(layer_sizes)-1:
            label = "OUTPUT"
        else:
            label = f"HIDDEN {i}"

        ax.text(
            x_positions[i],
            0.0,
            label,
            ha="center",
            color="#55D6C7",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")

    plt.tight_layout()

    return fig


def prepare_data(test_size):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:10px 0 25px 0;">
        <div style="font-size:12px;color:#50D4C5;
        letter-spacing:2px;font-weight:700;">
        MAI511-2
        </div>
        <div style="font-size:23px;font-weight:800;
        font-family:Manrope;color:#F1F5F9;">
        NEURAL LAB
        </div>
        <div style="font-size:12px;color:#91A6BB;">
        Advanced Machine Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "PROJECT NAVIGATION",
        [
            "01  Project Overview",
            "02  Dataset Exploration",
            "03  Data Preprocessing",
            "04  Network Architecture",
            "05  Model Training",
            "06  Model Evaluation",
            "07  Self-Learning Experiments",
            "08  Conclusion & Viva"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="font-size:12px;color:#8EA2B8;">
        <b>PROJECT</b><br>
        Multilayer Perceptron<br><br>
        <b>DATASET</b><br>
        Breast Cancer Wisconsin<br><br>
        <b>TASK</b><br>
        Binary Classification<br><br>
        <b>STUDENT</b><br>
        I MSc AIML
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.session_state.trained:
        st.success("Model trained")
    else:
        st.info("Model awaiting training")


# ============================================================
# 7. COMMON PAGE HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">ADVANCED MACHINE LEARNING · LAB EXERCISE 06</div>
    <div class="hero-title">Neural Diagnostics</div>
    <div class="hero-subtitle">
        Implementation and analysis of a Multilayer Perceptron
        using Artificial Neural Networks
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 01: PROJECT OVERVIEW
# ============================================================

if page == "01  Project Overview":

    st.markdown('<div class="section-label">01 / INTRODUCTION</div>',
                unsafe_allow_html=True)

    st.title("Project Overview")

    st.write(
        "This project implements a Multilayer Perceptron (MLP) "
        "for binary classification using the Breast Cancer Wisconsin "
        "Diagnostic dataset."
    )

    st.write(
        "The objective is to design, train and evaluate a neural "
        "network while understanding its architecture, activation "
        "functions and trainable parameters."
    )

    st.markdown("### Dataset at a glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Observations", f"{len(X):,}")
    c2.metric("Input features", X.shape[1])
    c3.metric("Output classes", 2)
    c4.metric("Missing values", int(X.isnull().sum().sum()))

    st.markdown("---")

    st.markdown("### Problem statement")

    st.info(
        "Develop an Artificial Neural Network using a Multilayer "
        "Perceptron to classify tumour observations into malignant "
        "and benign categories."
    )

    st.markdown("### Project objectives")

    objectives = [
        "Select and understand a real-world classification dataset.",
        "Identify input features and the target variable.",
        "Perform feature scaling and train-test splitting.",
        "Design an MLP with suitable hidden and output layers.",
        "Compile and train the neural network.",
        "Calculate and analyse trainable parameters.",
        "Evaluate model performance using classification metrics.",
        "Compare different activation functions and architectures."
    ]

    for item in objectives:
        st.markdown(f"- {item}")

    st.markdown("---")

    st.markdown("### Technology stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown(
            '<div class="info-card"><div class="info-title">'
            'DEEP LEARNING</div><div class="info-body">'
            'TensorFlow and Keras<br>Multilayer Perceptron'
            '</div></div>',
            unsafe_allow_html=True
        )

    with tech2:
        st.markdown(
            '<div class="info-card"><div class="info-title">'
            'DATA PROCESSING</div><div class="info-body">'
            'Pandas and NumPy<br>Scikit-learn'
            '</div></div>',
            unsafe_allow_html=True
        )

    with tech3:
        st.markdown(
            '<div class="info-card"><div class="info-title">'
            'VISUALIZATION</div><div class="info-body">'
            'Matplotlib<br>Seaborn and Streamlit'
            '</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### Workflow")

    st.write(
        "Dataset Selection → Exploration → Preprocessing → "
        "Architecture Design → Training → Parameter Analysis → "
        "Evaluation → Experiments"
    )


# ============================================================
# PAGE 02: DATASET EXPLORATION
# ============================================================

elif page == "02  Dataset Exploration":

    st.markdown('<div class="section-label">02 / DATA UNDERSTANDING</div>',
                unsafe_allow_html=True)

    st.title("Dataset Exploration")

    st.write(
        "The Breast Cancer Wisconsin Diagnostic dataset contains "
        "measurements computed from digitized images of fine needle "
        "aspirates of breast masses."
    )

    st.markdown("### Dataset information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total records", len(X))
        st.metric("Total features", X.shape[1])

    with col2:
        st.metric("Malignant cases", int((y == 0).sum()))
        st.metric("Benign cases", int((y == 1).sum()))

    st.markdown("### Target variable distribution")

    target_counts = y.map({
        0: "Malignant",
        1: "Benign"
    }).value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#142238")
    ax.set_facecolor("#142238")

    ax.bar(
        target_counts.index,
        target_counts.values,
        color=["#D69A58", "#16A6A0"],
        width=0.55
    )

    ax.set_ylabel("Number of observations", color="white")
    ax.set_title("Class Distribution", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#40546A")

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("### Feature information")

    st.dataframe(
        X.describe().T.round(3),
        width="stretch"
    )

    st.markdown("### Raw dataset preview")

    st.dataframe(
        X.head(10),
        width="stretch"
    )

    st.markdown("### Feature correlation analysis")

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#142238")

    sns.heatmap(
        X.corr(),
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.1
    )

    ax.set_title(
        "Correlation Between Input Features",
        color="white"
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.caption(
        "Correlation indicates the strength of linear association "
        "between numerical features."
    )


# ============================================================
# PAGE 03: PREPROCESSING
# ============================================================

elif page == "03  Data Preprocessing":

    st.markdown('<div class="section-label">03 / DATA PREPARATION</div>',
                unsafe_allow_html=True)

    st.title("Data Preprocessing")

    st.write(
        "Preprocessing prepares the dataset for neural network "
        "training and ensures that the model receives consistent "
        "numerical inputs."
    )

    st.markdown("### 1. Missing value analysis")

    missing = X.isnull().sum()

    st.metric("Total missing feature values", int(missing.sum()))

    if missing.sum() == 0:
        st.success(
            "No missing values were found. Imputation is therefore "
            "not required for this dataset."
        )

    st.markdown("### 2. Feature and target identification")

    st.code(
        "X = dataset.data\n"
        "y = dataset.target\n\n"
        "Input features = 30 numerical variables\n"
        "Target = diagnosis (0 or 1)"
    )

    st.markdown("### 3. Train-test splitting")

    st.write(
        "The dataset is divided into 80% training data and 20% "
        "testing data. Stratification preserves the original "
        "class proportions."
    )

    st.code(
        "train_test_split(\n"
        "    X, y,\n"
        "    test_size=0.20,\n"
        "    random_state=42,\n"
        "    stratify=y\n"
        ")"
    )

    st.markdown("### 4. Feature scaling")

    st.write(
        "StandardScaler transforms the features to have approximately "
        "zero mean and unit variance. This helps prevent features "
        "with larger numerical ranges from dominating optimization."
    )

    st.latex(r"z = \\frac{x-\\mu}{\\sigma}")

    st.code(
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_test_scaled = scaler.transform(X_test)"
    )

    st.warning(
        "The scaler is fitted only on the training set. The testing "
        "set is transformed using the same scaler to prevent "
        "data leakage."
    )

    st.markdown("### Preprocessing summary")

    preprocessing = pd.DataFrame({
        "Operation": [
            "Missing value handling",
            "Categorical encoding",
            "Feature scaling",
            "Train-test split",
            "Stratification"
        ],
        "Method": [
            "Not required",
            "Not required",
            "StandardScaler",
            "80% / 20%",
            "stratify=y"
        ],
        "Purpose": [
            "Dataset has no missing values",
            "All input features are numeric",
            "Normalize feature scales",
            "Evaluate unseen observations",
            "Preserve class distribution"
        ]
    })

    st.dataframe(preprocessing, width="stretch")


# ============================================================
# PAGE 04: NETWORK ARCHITECTURE
# ============================================================

elif page == "04  Network Architecture":

    st.markdown('<div class="section-label">04 / MODEL DESIGN</div>',
                unsafe_allow_html=True)

    st.title("Network Architecture")

    st.write(
        "The MLP is a feed-forward neural network consisting of "
        "an input layer, two hidden Dense layers and one output layer."
    )

    st.markdown("### Selected architecture")

    st.markdown("""
    <div class="info-card">
        <div class="info-title">BASELINE NETWORK</div>
        <div class="info-body">
            Input: 30 neurons<br>
            Hidden layer 1: 16 neurons – ReLU<br>
            Hidden layer 2: 8 neurons – ReLU<br>
            Output: 1 neuron – Sigmoid
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig = create_architecture(30, [16, 8])

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("### Layer-wise explanation")

    architecture = pd.DataFrame({
        "Layer": [
            "Input",
            "Hidden 1",
            "Hidden 2",
            "Output"
        ],
        "Neurons": [30, 16, 8, 1],
        "Activation": [
            "None",
            "ReLU",
            "ReLU",
            "Sigmoid"
        ],
        "Purpose": [
            "Receives input features",
            "Learns feature patterns",
            "Learns higher-level patterns",
            "Produces class probability"
        ]
    })

    st.dataframe(architecture, width="stretch")

    st.markdown("### Activation function justification")

    tab1, tab2, tab3 = st.tabs([
        "ReLU",
        "Sigmoid",
        "Why these choices?"
    ])

    with tab1:

        st.latex(r"f(x)=\\max(0,x)")

        st.write(
            "ReLU is used in the hidden layers because it introduces "
            "non-linearity and is computationally simple. It also "
            "helps reduce the vanishing gradient problem compared "
            "with sigmoid in many hidden-layer settings."
        )

    with tab2:

        st.latex(r"f(x)=\\frac{1}{1+e^{-x}}")

        st.write(
            "Sigmoid produces an output between 0 and 1. Since "
            "this is binary classification, the output represents "
            "the predicted probability of the benign class."
        )

    with tab3:

        st.write(
            "ReLU is selected for hidden layers to learn complex "
            "patterns. Sigmoid is selected for the single output "
            "neuron because the task involves two classes."
        )


# ============================================================
# PAGE 05: MODEL TRAINING
# ============================================================

elif page == "05  Model Training":

    st.markdown('<div class="section-label">05 / MODEL IMPLEMENTATION</div>',
                unsafe_allow_html=True)

    st.title("Model Training")

    st.write(
        "Configure and train the neural network. The default "
        "configuration follows the architecture described in "
        "the lab implementation."
    )

    st.markdown("### Training configuration")

    col1, col2 = st.columns(2)

    with col1:

        hidden_input = st.text_input(
            "Hidden layer neurons",
            value="16,8"
        )

        activation = st.selectbox(
            "Hidden layer activation",
            ["relu", "tanh", "sigmoid"]
        )

        learning_rate = st.selectbox(
            "Learning rate",
            [0.0001, 0.0005, 0.001, 0.005, 0.01],
            index=2
        )

    with col2:

        epochs = st.slider(
            "Maximum epochs",
            20, 200, 100, 10
        )

        batch_size = st.selectbox(
            "Batch size",
            [8, 16, 32, 64],
            index=2
        )

        test_size = st.select_slider(
            "Test data proportion",
            options=[0.15, 0.20, 0.25, 0.30, 0.35],
            value=0.20
        )

    dropout = st.slider(
        "Dropout rate",
        0.0, 0.5, 0.0, 0.05
    )

    use_early = st.checkbox(
        "Enable early stopping",
        value=True
    )

    st.markdown("---")

    try:

        hidden_layers = [
            int(v.strip())
            for v in hidden_input.split(",")
            if v.strip()
        ]

        if not hidden_layers or any(v <= 0 for v in hidden_layers):
            raise ValueError

    except ValueError:

        st.error(
            "Enter valid positive integers, e.g. 16,8"
        )
        st.stop()

    st.markdown("### Architecture preview")

    fig = create_architecture(
        X.shape[1],
        hidden_layers
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("### Compilation settings")

    st.code(
        f"Optimizer: Adam\n"
        f"Learning rate: {learning_rate}\n"
        f"Loss: Binary Crossentropy\n"
        f"Metrics: Accuracy\n"
        f"Epochs: {epochs}\n"
        f"Batch size: {batch_size}"
    )

    if st.button(
        "Train Neural Network",
        type="primary",
        width="stretch"
    ):

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

        (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            scaler
        ) = prepare_data(test_size)

        model = build_model(
            X.shape[1],
            hidden_layers,
            activation,
            dropout
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        callbacks = []

        if use_early:

            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=12,
                    restore_best_weights=True
                )
            )

        progress = st.progress(0)
        status = st.empty()

        with st.spinner("Training the MLP model..."):

            history = model.fit(
                X_train_scaled,
                y_train,
                validation_split=0.20,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )

            progress.progress(100)
            status.success("Training completed successfully.")

        probabilities = model.predict(
            X_test_scaled,
            verbose=0
        ).ravel()

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        metrics = {
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                y_test, probabilities
            )
        }

        summary = []

        model.summary(
            print_fn=lambda line: summary.append(line)
        )

        st.session_state.model_results = {
            "model": model,
            "history": history.history,
            "metrics": metrics,
            "predictions": predictions,
            "probabilities": probabilities,
            "y_test": np.array(y_test),
            "X_test": X_test_scaled,
            "X_train": X_train_scaled,
            "y_train": np.array(y_train),
            "scaler": scaler,
            "summary": "\n".join(summary),
            "hidden_layers": hidden_layers,
            "activation": activation,
            "dropout": dropout,
            "test_size": test_size
        }

        st.session_state.trained = True

    if st.session_state.trained:

        results = st.session_state.model_results

        st.success("A trained model is available.")

        st.markdown("### Training results")

        cols = st.columns(5)

        for col, (name, value) in zip(
            cols,
            results["metrics"].items()
        ):

            col.metric(
                name,
                f"{value:.4f}"
            )

        st.markdown("### Training history")

        hist = pd.DataFrame(results["history"])

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(hist["loss"], label="Training Loss")
        ax.plot(hist["val_loss"], label="Validation Loss")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Binary Crossentropy")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            hist["accuracy"],
            label="Training Accuracy"
        )

        ax.plot(
            hist["val_accuracy"],
            label="Validation Accuracy"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.caption(
            "Go to Model Evaluation to inspect the complete "
            "test-set performance."
        )


# ============================================================
# PAGE 06: MODEL EVALUATION
# ============================================================

elif page == "06  Model Evaluation":

    st.markdown('<div class="section-label">06 / PERFORMANCE ANALYSIS</div>',
                unsafe_allow_html=True)

    st.title("Model Evaluation")

    if not st.session_state.trained:

        st.warning(
            "Train the model first in the Model Training section."
        )

        st.stop()

    results = st.session_state.model_results

    st.markdown("### Test-set performance")

    cols = st.columns(5)

    for col, (name, value) in zip(
        cols,
        results["metrics"].items()
    ):

        col.metric(
            name,
            f"{value:.4f}"
        )

    st.markdown("---")

    st.markdown("### Confusion matrix")

    cm = confusion_matrix(
        results["y_test"],
        results["predictions"],
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("### Classification report")

    report = classification_report(
        results["y_test"],
        results["predictions"],
        target_names=["Malignant", "Benign"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).T.round(4),
        width="stretch"
    )

    st.markdown("### Training and validation curves")

    history = results["history"]

    c1, c2 = st.columns(2)

    with c1:

        fig, ax = plt.subplots()

        ax.plot(history["loss"], label="Training")
        ax.plot(history["val_loss"], label="Validation")

        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with c2:

        fig, ax = plt.subplots()

        ax.plot(
            history["accuracy"],
            label="Training"
        )

        ax.plot(
            history["val_accuracy"],
            label="Validation"
        )

        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("### Interpretation")

    train_loss = history["loss"][-1]
    val_loss = history["val_loss"][-1]

    if val_loss > train_loss * 1.25:

        st.warning(
            "The validation loss is considerably higher than "
            "training loss. This may indicate overfitting."
        )

    elif train_loss > 0.5 and val_loss > 0.5:

        st.warning(
            "Both training and validation losses remain high. "
            "This may indicate underfitting."
        )

    else:

        st.success(
            "The final training and validation losses are "
            "reasonably close. Inspect the complete curves "
            "to assess generalization."
        )

    st.write(
        "Accuracy indicates overall correctness. Precision, "
        "recall and F1-score provide additional information "
        "about class-wise prediction performance."
    )

    st.markdown("### Mathematical parameter analysis")

    param_df = calculate_parameters(
        X.shape[1],
        results["hidden_layers"]
    )

    st.dataframe(
        param_df,
        width="stretch"
    )

    total_parameters = int(
        param_df["Total Parameters"].sum()
    )

    st.metric(
        "Calculated trainable parameters",
        f"{total_parameters:,}"
    )

    model = results["model"]

    st.write(
        "**Keras total parameters:**",
        f"{model.count_params():,}"
    )

    trainable = sum(
        np.prod(w.shape)
        for w in model.trainable_weights
    )

    non_trainable = sum(
        np.prod(w.shape)
        for w in model.non_trainable_weights
    )

    st.write("**Trainable parameters:**", f"{trainable:,}")
    st.write("**Non-trainable parameters:**", f"{non_trainable:,}")

    st.markdown("### Model summary")

    st.code(results["summary"])


# ============================================================
# PAGE 07: SELF-LEARNING EXPERIMENTS
# ============================================================

elif page == "07  Self-Learning Experiments":

    st.markdown('<div class="section-label">07 / EXTENDED LEARNING</div>',
                unsafe_allow_html=True)

    st.title("Self-Learning Experiments")

    st.write(
        "Additional experiments are performed to understand "
        "how activation functions, network architecture and "
        "regularization influence MLP performance."
    )

    st.info(
        "These experiments extend the basic implementation "
        "and help analyse model behaviour under different "
        "configurations."
    )

    if not st.session_state.trained:

        st.warning(
            "Train the baseline model first in Model Training."
        )

        st.stop()

    results = st.session_state.model_results

    X_train_scaled = results["X_train"]
    X_test_scaled = results["X_test"]
    y_train = results["y_train"]
    y_test = results["y_test"]

    st.markdown("### Experiment 1: Activation function comparison")

    st.write(
        "ReLU, tanh and sigmoid are compared in the hidden layers "
        "while keeping the data split, layer sizes and other "
        "training settings consistent."
    )

    if st.button(
        "Run activation comparison",
        key="activation_test"
    ):

        comparison = []

        for activation in ["relu", "tanh", "sigmoid"]:

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(
                    results["hidden_layers"][0],
                    activation=activation
                ),
                *[
                    tf.keras.layers.Dense(
                        units,
                        activation=activation
                    )
                    for units in results["hidden_layers"][1:]
                ],
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid"
                )
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.001
                ),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            model.fit(
                X_train_scaled,
                y_train,
                validation_split=0.20,
                epochs=60,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=8,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )

            probabilities = model.predict(
                X_test_scaled,
                verbose=0
            ).ravel()

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            comparison.append({
                "Activation": activation,
                "Accuracy": accuracy_score(
                    y_test, predictions
                ),
                "Precision": precision_score(
                    y_test, predictions,
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions,
                    zero_division=0
                ),
                "F1-score": f1_score(
                    y_test, predictions,
                    zero_division=0
                ),
                "ROC-AUC": roc_auc_score(
                    y_test, probabilities
                )
            })

        comparison_df = pd.DataFrame(comparison)

        st.session_state.experiment_results = comparison_df

    if st.session_state.experiment_results is not None:

        st.dataframe(
            st.session_state.experiment_results.round(4),
            width="stretch"
        )

        fig, ax = plt.subplots(figsize=(9, 4))

        st.session_state.experiment_results.set_index(
            "Activation"
        )[["Accuracy", "F1-score", "ROC-AUC"]].plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Activation Function Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("---")

    st.markdown("### Experiment 2: Dropout regularization")

    st.write(
        "Dropout randomly disables a fraction of neurons "
        "during training. This can reduce co-adaptation and "
        "help prevent overfitting."
    )

    st.latex(
        r"\\text{Dropout: randomly set selected activations to zero}"
    )

    st.write(
        "Compare a baseline network with a network containing "
        "Dropout(0.25). Both models use the same training and "
        "testing data."
    )

    if st.button(
        "Run dropout experiment",
        key="dropout_test"
    ):

        dropout_results = []

        for name, dropout_rate in [
            ("Baseline", 0.0),
            ("Dropout 0.25", 0.25)
        ]:

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])

            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            history = model.fit(
                X_train_scaled,
                y_train,
                validation_split=0.20,
                epochs=100,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=12,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )

            probabilities = model.predict(
                X_test_scaled,
                verbose=0
            ).ravel()

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            dropout_results.append({
                "Model": name,
                "Test Accuracy": accuracy_score(
                    y_test, predictions
                ),
                "Test F1-score": f1_score(
                    y_test, predictions,
                    zero_division=0
                ),
                "Epochs": len(history.history["loss"]),
                "Final Training Loss": history.history["loss"][-1],
                "Final Validation Loss": history.history["val_loss"][-1]
            })

        dropout_df = pd.DataFrame(dropout_results)

        st.dataframe(
            dropout_df.round(4),
            width="stretch"
        )

        st.bar_chart(
            dropout_df.set_index("Model")[
                ["Test Accuracy", "Test F1-score"]
            ]
        )

    st.markdown("---")

    st.markdown("### Experiment 3: Architecture comparison")

    st.write(
        "A shallow network and a deeper network are compared "
        "to study how the number of hidden layers and neurons "
        "affect learning."
    )

    if st.button(
        "Run architecture comparison",
        key="architecture_test"
    ):

        architectures = {
            "Shallow (8)": [8],
            "Baseline (16, 8)": [16, 8],
            "Deeper (32, 16, 8)": [32, 16, 8]
        }

        architecture_results = []

        for name, layers in architectures.items():

            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(42)

            model = tf.keras.Sequential()

            model.add(
                tf.keras.layers.Input(
                    shape=(X.shape[1],)
                )
            )

            for units in layers:

                model.add(
                    tf.keras.layers.Dense(
                        units,
                        activation="relu"
                    )
                )

            model.add(
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid"
                )
            )

            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            model.fit(
                X_train_scaled,
                y_train,
                validation_split=0.20,
                epochs=60,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=8,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )

            probabilities = model.predict(
                X_test_scaled,
                verbose=0
            ).ravel()

            predictions = (
                probabilities >= 0.5
            ).astype(int)

            architecture_results.append({
                "Architecture": name,
                "Parameters": model.count_params(),
                "Accuracy": accuracy_score(
                    y_test, predictions
                ),
                "F1-score": f1_score(
                    y_test, predictions,
                    zero_division=0
                )
            })

        architecture_df = pd.DataFrame(
            architecture_results
        )

        st.dataframe(
            architecture_df.round(4),
            width="stretch"
        )

        st.bar_chart(
            architecture_df.set_index("Architecture")[
                ["Accuracy", "F1-score"]
            ]
        )

    st.markdown("---")

    st.markdown("### What these experiments demonstrate")

    st.markdown("""
    - **Activation comparison:** Different activation functions
      can change how effectively the network learns patterns.
    - **Dropout:** Regularization can help control overfitting,
      although it may also reduce performance if applied too strongly.
    - **Architecture comparison:** Increasing network depth or
      width changes model capacity and trainable parameter count.
    """)


# ============================================================
# PAGE 08: CONCLUSION AND VIVA
# ============================================================

elif page == "08  Conclusion & Viva":

    st.markdown('<div class="section-label">08 / PROJECT SUMMARY</div>',
                unsafe_allow_html=True)

    st.title("Conclusion & Viva Preparation")

    st.markdown("### Conclusion")

    st.write(
        "The experiment demonstrates the implementation of a "
        "Multilayer Perceptron for binary classification using "
        "the Breast Cancer Wisconsin Diagnostic dataset."
    )

    st.write(
        "The workflow includes dataset exploration, preprocessing, "
        "feature scaling, architecture design, model compilation, "
        "training, mathematical parameter calculation and "
        "performance evaluation."
    )

    st.write(
        "Additional experiments investigate activation functions, "
        "dropout regularization and network architecture. These "
        "experiments provide insight into how neural network "
        "design choices influence model performance."
    )

    if st.session_state.trained:

        results = st.session_state.model_results

        st.markdown("### Actual model results")

        cols = st.columns(4)

        for col, metric in zip(
            cols,
            ["Accuracy", "Precision", "Recall", "F1-score"]
        ):

            col.metric(
                metric,
                f"{results['metrics'][metric]:.4f}"
            )

    else:

        st.info(
            "Train the model to display your actual performance results."
        )

    st.markdown("---")

    st.markdown("### Viva preparation")

    questions = [
        (
            "1. What is a Multilayer Perceptron?",
            "An MLP is a feed-forward artificial neural network "
            "consisting of an input layer, one or more hidden "
            "layers and an output layer."
        ),
        (
            "2. Why is this a classification problem?",
            "The target variable contains two discrete classes: "
            "malignant and benign."
        ),
        (
            "3. Why is feature scaling required?",
            "Scaling puts features on comparable numerical scales "
            "and helps gradient-based optimization."
        ),
        (
            "4. Why is ReLU used in hidden layers?",
            "ReLU introduces non-linearity and is computationally "
            "simple. It is commonly used in hidden layers."
        ),
        (
            "5. Why is sigmoid used in the output layer?",
            "Sigmoid generates an output between 0 and 1, which "
            "can represent the probability of the positive class."
        ),
        (
            "6. Why is binary cross-entropy used?",
            "It measures the difference between the actual binary "
            "labels and the predicted probabilities."
        ),
        (
            "7. What is an optimizer?",
            "An optimizer updates the network weights and biases "
            "to minimize the loss function."
        ),
        (
            "8. What is Adam?",
            "Adam is an adaptive optimization algorithm that uses "
            "estimates of first and second moments of gradients."
        ),
        (
            "9. What are trainable parameters?",
            "Trainable parameters are the weights and biases "
            "that the model learns during training."
        ),
        (
            "10. How are Dense layer parameters calculated?",
            "Parameters = (input neurons × output neurons) + "
            "output neurons, where the final term represents biases."
        ),
        (
            "11. What is overfitting?",
            "Overfitting occurs when the model learns training "
            "data patterns too specifically and performs worse "
            "on unseen data."
        ),
        (
            "12. What is dropout?",
            "Dropout randomly disables a fraction of neurons "
            "during training to help reduce overfitting."
        ),
        (
            "13. What is early stopping?",
            "Early stopping monitors validation performance and "
            "stops training when improvement stalls."
        ),
        (
            "14. Why use a confusion matrix?",
            "It shows the number of correct and incorrect "
            "predictions for each class."
        ),
        (
            "15. What is the F1-score?",
            "The F1-score is the harmonic mean of precision "
            "and recall."
        ),
        (
            "16. What is data leakage?",
            "Data leakage occurs when information from the test "
            "data improperly influences model training or preprocessing."
        )
    ]

    for question, answer in questions:

        with st.expander(question):

            st.write(answer)

    st.markdown("---")

    st.markdown("### Final project checklist")

    st.markdown("""
    - Dataset selected and explored.
    - Input features and target identified.
    - Preprocessing and scaling performed.
    - Training and testing datasets created.
    - MLP architecture designed.
    - Activation functions justified.
    - Model compiled and trained.
    - Model summary analysed.
    - Trainable parameters calculated mathematically.
    - Classification metrics evaluated.
    - Training and validation curves visualized.
    - Additional self-learning experiments conducted.
    """)

    st.markdown("---")

    st.caption(
        "MAI511-2 | Advanced Machine Learning | "
        "Lab Exercise 6 | I MSc AIML"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#71869B;font-size:12px;">
    NEURAL DIAGNOSTICS · MULTILAYER PERCEPTRON<br>
    MAI511-2 ADVANCED MACHINE LEARNING
    </div>
    """,
    unsafe_allow_html=True
)
