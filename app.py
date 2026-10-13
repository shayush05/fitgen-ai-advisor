import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

st.set_page_config(
    page_title="FitGen AI - Nutrition & Fitness Advisor",
    page_icon="💪",
    layout="wide"
)

# --- 1. SIDEBAR: USER METRICS & GOALS ---
st.sidebar.header("👤 Athlete Profile")
age = st.sidebar.slider("Age", min_value=15, max_value=80, value=21)
gender = st.sidebar.selectbox("Biological Sex", ["Male", "Female"])
weight = st.sidebar.number_input("Weight (kg)", min_value=35.0, max_value=180.0, value=75.0, step=0.5)
height = st.sidebar.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=175.0, step=1.0)

activity_level = st.sidebar.selectbox(
    "Activity Level",
    [
        "Sedentary (desk job, minimal exercise)",
        "Lightly Active (training 1-3 days/week)",
        "Moderately Active (training 3-5 days/week)",
        "Very Active (heavy training 6-7 days/week)"
    ]
)

goal = st.sidebar.selectbox(
    "Primary Goal",
    [
        "Fat Loss (Caloric Deficit)",
        "Maintenance / Body Recomposition",
        "Muscle Growth (Caloric Surplus)"
    ]
)

# --- 2. PRECISE NUTRITION ENGINE (Mifflin-St Jeor) ---
def compute_nutrition_plan(w, h, a, sex, act, target_goal):
    # Basal Metabolic Rate (BMR)
    if sex == "Male":
        bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
    else:
        bmr = (10 * w) + (6.25 * h) - (5 * a) - 161

    # Total Daily Energy Expenditure (TDEE) multiplier
    multipliers = {
        "Sedentary (desk job, minimal exercise)": 1.2,
        "Lightly Active (training 1-3 days/week)": 1.375,
        "Moderately Active (training 3-5 days/week)": 1.55,
        "Very Active (heavy training 6-7 days/week)": 1.725
    }
    tdee = bmr * multipliers[act]

    # Target calories and macronutrient splits
    if "Fat Loss" in target_goal:
        target_calories = tdee - 450
        protein_g = w * 2.2  # Higher protein to protect lean mass
    elif "Muscle Growth" in target_goal:
        target_calories = tdee + 350
        protein_g = w * 1.9  # Hypertrophy target
    else:
        target_calories = tdee
        protein_g = w * 1.8  # Body recomposition baseline

    # Fat: 25% of total caloric intake (9 kcal per gram)
    fat_g = (target_calories * 0.25) / 9

    # Carbohydrates: Remaining calories (4 kcal per gram)
    carbs_kcal = target_calories - ((protein_g * 4) + (fat_g * 9))
    carbs_g = max(50.0, carbs_kcal / 4)

    return int(target_calories), int(protein_g), int(fat_g), int(carbs_g)

target_cals, target_protein, target_fats, target_carbs = compute_nutrition_plan(
    weight, height, age, gender, activity_level, goal
)

# --- 3. UI DASHBOARD: TARGETS DISPLAY ---
st.title("💪 FitGen AI - Personalized Nutrition Advisor")
st.caption("Hybrid sports science calculator paired with RAG retrieval and a fine-tuned LoRA intelligence engine.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Daily Calorie Target", f"{target_cals} kcal")
col2.metric("Protein", f"{target_protein} g", help="Key for muscle protein synthesis and recovery")
col3.metric("Carbohydrates", f"{target_carbs} g", help="Main energy source for anaerobic training")
col4.metric("Healthy Fats", f"{target_fats} g", help="Essential for hormone regulation and joint health")

st.divider()

# --- 4. BACKEND: RAG & PEFT MODEL INITIALIZATION ---
@st.cache_resource
def load_rag_and_model():
    # Detect hardware environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load Base Flan-T5
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=precision)

    # Attach LoRA weights if available, otherwise fallback cleanly to base model
    weights_path = "./fitgen_final_weights"
    if os.path.exists(weights_path):
        model = PeftModel.from_pretrained(base_model, weights_path)
    else:
        model = base_model

    model.to(device)
    model.eval()

    # Load Embedding Model & ChromaDB
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    db_client = chromadb.Client()
    collection = db_client.get_or_create_collection(name="gym_knowledge")

    # Seed ChromaDB if fresh/empty (guarantees zero missing-context crashes in cloud deployment)
    if collection.count() == 0:
        curated_facts = [
            "For fat loss, a caloric deficit of 300 to 500 calories below maintenance supports steady fat reduction while preserving lean mass.",
            "For muscle hypertrophy during a bulk, a slight surplus of 250 to 400 calories above maintenance optimizes growth while limiting fat gain.",
            "Daily protein intake for resistance training should range between 1.6g and 2.2g per kilogram of total body weight.",
            "Dietary fat should account for 20% to 30% of total daily calories to regulate hormone production, including testosterone.",
            "Carbohydrates are the primary fuel source for anaerobic glycolysis; consume 3g to 7g per kilogram depending on training volume.",
            "Body recomposition is achievable at maintenance calories with high protein (2.0g-2.4g/kg) and progressive overload training.",
            "Iron and Vitamin C should be consumed together to maximize non-heme iron absorption and maintain red blood cell oxygen transport.",
            "Zinc and Magnesium play critical roles in muscle recovery, protein synthesis, and central nervous system regulation during deep sleep.",
            "Creatine monohydrate supplementation at 3g to 5g daily saturates intramuscular phosphocreatine stores, enhancing peak strength.",
            "Adequate hydration directly impacts intracellular signaling; consume 35ml to 45ml of water per kilogram of body weight daily."
        ]
        embeddings = [embed_model.encode(fact).tolist() for fact in curated_facts]
        ids = [f"fact_{i}" for i in range(len(curated_facts))]
        collection.add(documents=curated_facts, embeddings=embeddings, ids=ids)

    return tokenizer, model, embed_model, collection, device

with st.spinner("Initializing models and knowledge engine..."):
    tokenizer, model, embed_model, collection, device = load_rag_and_model()

# --- 5. ADVISOR INTERFACE ---
st.subheader("💬 Ask Your Nutrition Advisor")
if "Fat Loss" in goal:
    default_query = "How many calories should I cut to lose fat?"
elif "Muscle Growth" in goal:
    default_query = "How much should I eat to bulk up cleanly?"
else:
    default_query = "How can I achieve body recomposition?"
user_question = st.text_input("Enter your question regarding meals, macros, recovery, or micronutrients:", value=default_query)

if st.button("Generate Coaching Strategy", type="primary"):
    if user_question.strip():
        with st.spinner("Analyzing profile and retrieving sports nutrition guidelines..."):
            # 1. RAG retrieval
            q_embedding = embed_model.encode(user_question).tolist()
            search_results = collection.query(query_embeddings=[q_embedding], n_results=1)

            retrieved_fact = (
                search_results["documents"][0][0]
                if search_results.get("documents") and search_results["documents"][0]
                else "Focus on whole protein sources, healthy dietary fats, complex carbohydrates, and proper recovery."
            )

            # 2. Formulate context-rich prompt
            prompt = f"Use this fact to answer the question.\nFact: {retrieved_fact}\nQuestion: {user_question}\nAnswer:"

            # 3. Model inference
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=4,             # Explores the 4 most logical paths to build a coherent sentence
                    early_stopping=True,     # Stops generating once it finishes a natural sentence
                    repetition_penalty=1.2   # Prevents it from stuttering or repeating the same word
                )

            ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 4. Display result
            st.success("🎯 Custom Coaching Strategy")
            st.write(ai_response)

            # 5. Diagnostic breakdown
            with st.expander("🔍 Nutritional Math & RAG Fact Reference"):
                st.markdown(f"**Retrieved Knowledge Anchor:** *{retrieved_fact}*")
                st.markdown(f"**Total Daily Energy Expenditure (Maintenance):** ~`{int(target_cals)}` kcal adjusted for `{goal}`")
                st.markdown(f"**Protein Split:** `{target_protein}g` (~`{round((target_protein * 4 / target_cals) * 100)}%` of calories)")
                st.markdown(f"**Carb Split:** `{target_carbs}g` (~`{round((target_carbs * 4 / target_cals) * 100)}%` of calories)")
                st.markdown(f"**Fat Split:** `{target_fats}g` (~`{round((target_fats * 9 / target_cals) * 100)}%` of calories)")
    else:
        st.warning("Please enter a question to generate nutrition coaching.")
