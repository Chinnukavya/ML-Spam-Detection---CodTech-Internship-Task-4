# ============================================================
#   SPAM EMAIL DETECTION - MACHINE LEARNING
#   CodTech Internship - Task 4
#   Run in VS Code: python spam_detection.py
# ============================================================

# ── STEP 1: Install libraries first (run in terminal) ────────
# pip install scikit-learn pandas numpy matplotlib seaborn

# ── STEP 2: Import Libraries ─────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("    📧 SPAM EMAIL DETECTION - ML MODEL")
print("       CodTech Internship - Task 4")
print("=" * 55)
print("✅ All libraries imported successfully!\n")


# ── STEP 3: Load Dataset ──────────────────────────────────────
print("📂 Loading dataset...")

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print(f"✅ Dataset loaded successfully!")
print(f"   Total records : {len(df)}")
print(f"   Shape         : {df.shape}")
print()


# ── STEP 4: Explore Data ──────────────────────────────────────
print("=" * 55)
print("         DATASET OVERVIEW")
print("=" * 55)

spam_count = (df['label'] == 'spam').sum()
ham_count  = (df['label'] == 'ham').sum()

print(f"\n📊 Total Messages  : {len(df)}")
print(f"🚨 Spam Messages   : {spam_count} ({spam_count/len(df)*100:.1f}%)")
print(f"✅ Ham Messages    : {ham_count}  ({ham_count/len(df)*100:.1f}%)")

print("\n📝 First 5 rows:")
print(df.head().to_string())

print("\n📝 Sample SPAM messages:")
for msg in df[df['label'] == 'spam']['message'].head(3):
    print(f"   → {msg[:80]}...")

print("\n📝 Sample HAM messages:")
for msg in df[df['label'] == 'ham']['message'].head(3):
    print(f"   → {msg[:80]}")
print()


# ── STEP 5: Visualize Data ────────────────────────────────────
print("📊 Generating visualizations...")

df['msg_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Spam Detection - Data Analysis', fontsize=16, fontweight='bold')

# Plot 1: Count bar chart
df['label'].value_counts().plot(
    kind='bar', ax=axes[0, 0],
    color=['#2196F3', '#F44336'], edgecolor='black', width=0.5
)
axes[0, 0].set_title('Spam vs Ham Count', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Label')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticklabels(['Ham ✅', 'Spam 🚨'], rotation=0)
for i, v in enumerate(df['label'].value_counts()):
    axes[0, 0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Plot 2: Pie chart
df['label'].value_counts().plot(
    kind='pie', ax=axes[0, 1],
    colors=['#2196F3', '#F44336'],
    autopct='%1.1f%%', startangle=90,
    explode=(0.05, 0.05)
)
axes[0, 1].set_title('Distribution %', fontsize=13, fontweight='bold')
axes[0, 1].set_ylabel('')

# Plot 3: Message length histogram
for label, color in [('ham', '#2196F3'), ('spam', '#F44336')]:
    df[df['label'] == label]['msg_length'].plot(
        kind='hist', bins=40, alpha=0.6,
        ax=axes[1, 0], color=color, label=label.upper()
    )
axes[1, 0].set_title('Message Length Distribution', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Message Length (characters)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# Plot 4: Word count boxplot
df.boxplot(column='word_count', by='label', ax=axes[1, 1], patch_artist=True)
axes[1, 1].set_title('Word Count by Label', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Label')
axes[1, 1].set_ylabel('Word Count')
plt.suptitle('Spam Detection - Data Analysis', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Data analysis chart saved as 'data_analysis.png'\n")


# ── STEP 6: Preprocess Data ───────────────────────────────────
print("⚙️  Preprocessing data...")

df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Data Split Complete!")
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}\n")


# ── STEP 7: TF-IDF Vectorization ─────────────────────────────
print("🔤 Applying TF-IDF Vectorization...")

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"✅ TF-IDF Vectorization Complete!")
print(f"   Training matrix shape : {X_train_tfidf.shape}")
print(f"   Testing matrix shape  : {X_test_tfidf.shape}\n")


# ── STEP 8: Train Naive Bayes Model ──────────────────────────
print("=" * 55)
print("     🤖 MODEL 1: NAIVE BAYES")
print("=" * 55)

nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train_tfidf, y_train)

nb_predictions = nb_model.predict(X_test_tfidf)
nb_accuracy    = accuracy_score(y_test, nb_predictions)

print(f"\n✅ Accuracy  : {nb_accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, nb_predictions,
                             target_names=['Ham', 'Spam']))


# ── STEP 9: Train Logistic Regression Model ───────────────────
print("=" * 55)
print("   🤖 MODEL 2: LOGISTIC REGRESSION")
print("=" * 55)

lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

lr_predictions = lr_model.predict(X_test_tfidf)
lr_accuracy    = accuracy_score(y_test, lr_predictions)

print(f"\n✅ Accuracy  : {lr_accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, lr_predictions,
                             target_names=['Ham', 'Spam']))


# ── STEP 10: Confusion Matrix ─────────────────────────────────
print("📉 Generating confusion matrix...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm_nb = confusion_matrix(y_test, nb_predictions)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_nb,
                                display_labels=['Ham', 'Spam'])
disp1.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Naive Bayes\nAccuracy: {nb_accuracy*100:.2f}%',
                  fontsize=13, fontweight='bold')

cm_lr = confusion_matrix(y_test, lr_predictions)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                                display_labels=['Ham', 'Spam'])
disp2.plot(ax=axes[1], colorbar=False, cmap='Greens')
axes[1].set_title(f'Logistic Regression\nAccuracy: {lr_accuracy*100:.2f}%',
                  fontsize=13, fontweight='bold')

plt.suptitle('Confusion Matrix Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved as 'confusion_matrix.png'\n")


# ── STEP 11: Model Comparison Chart ──────────────────────────
print("📊 Generating model comparison chart...")

models     = ['Naive Bayes', 'Logistic Regression']
accuracies = [nb_accuracy * 100, lr_accuracy * 100]
colors     = ['#2196F3', '#4CAF50']

plt.figure(figsize=(7, 5))
bars = plt.bar(models, accuracies, color=colors,
               edgecolor='black', width=0.4)
plt.ylim(90, 100)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Model', fontsize=12)

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f'{acc:.2f}%',
        ha='center', va='bottom',
        fontsize=13, fontweight='bold'
    )

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Model comparison chart saved as 'model_comparison.png'\n")


# ── STEP 12: Top Spam Keywords ────────────────────────────────
print("🔑 Generating top spam keywords chart...")

feature_names = tfidf.get_feature_names_out()
coefs         = lr_model.coef_[0]

top_spam_idx  = coefs.argsort()[-20:][::-1]
top_ham_idx   = coefs.argsort()[:20]

top_spam_words = [(feature_names[i], coefs[i]) for i in top_spam_idx]
top_ham_words  = [(feature_names[i], coefs[i]) for i in top_ham_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

spam_words, spam_scores = zip(*top_spam_words)
axes[0].barh(spam_words, spam_scores, color='#F44336', edgecolor='black')
axes[0].set_title('Top 20 SPAM Keywords', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Coefficient Score')
axes[0].invert_yaxis()

ham_words, ham_scores = zip(*top_ham_words)
axes[1].barh(ham_words, ham_scores, color='#2196F3', edgecolor='black')
axes[1].set_title('Top 20 HAM Keywords', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Coefficient Score')
axes[1].invert_yaxis()

plt.suptitle('Most Influential Words for Classification',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('top_keywords.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Keywords chart saved as 'top_keywords.png'\n")


# ── STEP 13: Test Custom Messages ────────────────────────────
def predict_spam(message, model=lr_model):
    msg_tfidf   = tfidf.transform([message])
    prediction  = model.predict(msg_tfidf)[0]
    probability = model.predict_proba(msg_tfidf)[0]
    label       = "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
    confidence  = max(probability) * 100
    short_msg   = message[:65] + "..." if len(message) > 65 else message
    print(f"  Message    : {short_msg}")
    print(f"  Result     : {label}")
    print(f"  Confidence : {confidence:.2f}%")
    print("  " + "-" * 50)

print("=" * 55)
print("      🧪 SPAM DETECTION - LIVE TEST")
print("=" * 55)
print()

test_messages = [
    "Congratulations! You won a FREE iPhone. Click here to claim now!",
    "Hey, are we still meeting tomorrow for lunch?",
    "URGENT: Your bank account has been suspended. Call now!",
    "Can you please send me the project report by evening?",
    "Win cash prizes worth $1000! Limited offer. Reply WIN now!",
    "Hi mom, I will be home late tonight. Do not wait for me.",
    "FREE entry to our prize draw! Text WIN to 87121 now!",
    "The meeting is rescheduled to 3 PM. Please confirm attendance."
]

for msg in test_messages:
    predict_spam(msg)


# ── STEP 14: Final Summary ────────────────────────────────────
print()
print("=" * 55)
print("        📋 FINAL EVALUATION SUMMARY")
print("=" * 55)

for name, preds in [('Naive Bayes', nb_predictions),
                     ('Logistic Regression', lr_predictions)]:
    acc  = accuracy_score(y_test, preds) * 100
    prec = precision_score(y_test, preds) * 100
    rec  = recall_score(y_test, preds) * 100
    f1   = f1_score(y_test, preds) * 100
    print(f"\n🤖 {name}")
    print(f"   Accuracy  : {acc:.2f}%")
    print(f"   Precision : {prec:.2f}%")
    print(f"   Recall    : {rec:.2f}%")
    print(f"   F1 Score  : {f1:.2f}%")

best = 'Naive Bayes' if nb_accuracy >= lr_accuracy else 'Logistic Regression'
print()
print("=" * 55)
print(f"  🏆 BEST MODEL : {best}")
print("=" * 55)

print("\n📁 Files saved in your project folder:")
print("   → data_analysis.png")
print("   → confusion_matrix.png")
print("   → model_comparison.png")
print("   → top_keywords.png")
print()
print("✅ Task 4 Complete! Push to GitHub now.")
print("=" * 55)


# ── STEP 15: Interactive Input ────────────────────────────────
print()
try:
    while True:
        user_input = input("\n🔤 Enter your own message to test (or 'quit' to exit): ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Exiting. Good luck with your internship!")
            break
        if user_input:
            print()
            predict_spam(user_input)
        else:
            print("Please enter a message.")
except KeyboardInterrupt:
    print("\n👋 Program stopped.")