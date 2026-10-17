import pickle
import re
import nltk
from nltk.corpus import stopwords

# Saved models को लोड करें
try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("❌ Error: 'vectorizer.pkl' या 'model.pkl' नहीं मिले। पहले main script चलाएं।")
    exit()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return " ".join([w for w in text.split() if w not in stop_words])

print("🤖 --- LIVE SPAM MESSAGING CLASSIIFIER --- 🤖")
user_input = input("👉 Apna text message yahan type karein: ")

cleaned_msg = clean_text(user_input)
vectorized_msg = vectorizer.transform([cleaned_msg])
prediction = model.predict(vectorized_msg)

print("\n-------------------------------------")
if prediction[0] == 1:
    print("🚨 RESULT: Yeh ek SPAM message hai!")
else:
    print("✅ RESULT: Yeh ek safe HAM message hai.")
print("-------------------------------------")
