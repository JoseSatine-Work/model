import streamlit as st
from functools import lru_cache
import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import coremltools
import json
import time

# Set page config first for better loading experience
st.set_page_config(
    page_title="Nutrition Analyzer",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize loading state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Show loading placeholder
loading_placeholder = st.empty()

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def initialize_gemini():
    """Initialize Gemini model with caching"""
    load_dotenv()
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_custom_model():
    """Load CoreML model with caching"""
    try:
        model = coremltools.models.MLModel('model/MyModel.mlmodel')
        with open('model/list.json', 'r') as f:
            nutrition_data = json.load(f)
        return model, nutrition_data
    except Exception as e:
        st.error(f"Failed to load custom model: {str(e)}")
        return None, None

# Initialize models with loading indicator
with loading_placeholder.container():
    with st.spinner('🚀 Loading AI models... This may take a moment...'):
        gemini_model = initialize_gemini()
        custom_model, nutrition_data = load_custom_model()
        
        if gemini_model is None or custom_model is None:
            st.error("Failed to initialize required models. Please check the error messages above.")
            st.stop()
            
        st.session_state.initialized = True

# Clear loading placeholder after initialization
if st.session_state.initialized:
    loading_placeholder.empty()

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def get_gemini_response(input_text, image, prompt):
    try:
        response = gemini_model.generate_content([input_text, image[0], prompt])
        return response.text
    except Exception as e:
        st.error(f"Error analyzing image with Gemini: {str(e)}")
        return None

def get_custom_model_response(image):
    try:
        # Resize image to match model's expected input size
        image = image.resize((224, 224))
        
        # Make prediction
        prediction = custom_model.predict({'input_1': image})
        
        # Define food classes
        classes = [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
            "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
            "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
            "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
            "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
            "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
            "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
            "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon",
            "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros",
            "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich",
            "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette",
            "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
            "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
            "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops",
            "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
            "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
            "tiramisu", "tuna_tartare", "waffles"
        ]
        
        max_index = prediction['linear_510'].argmax()
        predicted_class = classes[max_index]
        
        # Convert class name to JSON format (title case with spaces)
        json_class_name = predicted_class.replace('_', ' ').title()
        
        # Find matching nutrition data
        nutrition = next((item for item in nutrition_data if item["name"] == json_class_name), None)
        
        if nutrition:
            dv = nutrition["dailyValue"]
            result = {
                "predicted_food": json_class_name,
                "nutrition": {
                    "Total Fat": dv['totalFat'].split()[-1],
                    "Saturated Fat": dv['saturatedFat'].split()[-1],
                    "Sodium": dv['sodium'].split()[-1],
                    "Carbohydrates": dv['totalCarbohydrates'].split()[-1],
                    "Protein": dv['protein'].split()[0] + "g"
                }
            }
            return result
        else:
            return {"error": "Nutrition data not found for this dish"}
            
    except Exception as e:
        st.error(f"Error analyzing image with custom model: {str(e)}")
        return None

# Page title is already set at the top

st.header('Food Scan with Multiple Models')
st.subheader('Choose your analysis method:')

# Cache the input prompt to avoid redefining it on every run
@st.cache_data(ttl=3600)
def get_input_prompt():
    return """
    You have to identify different types of food in images. 
    The system should accurately detect and label various foods displayed in the image, providing the name 
    of the food and its location within the image (e.g., bottom left, right corner, etc.). Additionally, 
    the system should extract nutritional information and categorize the type of food (e.g., fruits, vegetables, grains, etc.) 
    based on the detected items. The output should include a comprehensive report or display showing the
    identified foods, their positions, names, and corresponding nutritional details.
    """

def main():
    # Create two columns for the options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Gemini API")
        st.markdown("Analyze food using Google's Gemini AI model")
        use_gemini = st.checkbox("Use Gemini API", value=True, key="gemini_checkbox")

    with col2:
        st.markdown("### Custom Model")
        st.markdown("Analyze food using your custom model")
        use_custom = st.checkbox("Use Custom Model", value=False, key="custom_checkbox")

    # Use session state to store the uploaded file
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    uploaded_file = st.file_uploader("Choose an image of the food or food table", 
                                    type=["jpg", 'jpeg', 'png'],
                                    key="file_uploader")
    
    # Update session state when a new file is uploaded
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    
    # Display the uploaded image
    image_display = st.empty()
    if st.session_state.uploaded_file is not None:
        try:
            image = Image.open(st.session_state.uploaded_file)
            image_display.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            st.session_state.uploaded_file = None
    
    user_input = st.text_input("Input prompt: ", key='user_input')
    submit = st.button("Scan the Food(s)", key="scan_button")

    if submit:
        if st.session_state.uploaded_file is None:
            st.error("Please upload an image first")
            return
            
        if not (use_gemini or use_custom):
            st.error("Please select at least one analysis method")
            return
            
        try:
            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process image once and reuse
            image_data = None
            if use_gemini or use_custom:
                status_text.text("Processing image...")
                progress_bar.progress(25)
                image_data = input_image_details(st.session_state.uploaded_file)
                progress_bar.progress(50)
            
            # Get responses
            if use_gemini:
                status_text.text("Analyzing with Gemini...")
                with st.expander("Gemini API Analysis", expanded=True):
                    with st.spinner("Gemini is analyzing the image..."):
                        gemini_response = get_gemini_response(get_input_prompt(), image_data, user_input)
                        if gemini_response:
                            st.markdown(gemini_response)
                progress_bar.progress(75)
            
            if use_custom:
                status_text.text("Analyzing with Custom Model...")
                with st.expander("Custom Model Analysis", expanded=True):
                    with st.spinner("Custom model is analyzing the image..."):
                        custom_response = get_custom_model_response(Image.open(st.session_state.uploaded_file))
                        if custom_response:
                            if "error" in custom_response:
                                st.error(custom_response["error"])
                            else:
                                st.subheader("Predicted Food:")
                                st.markdown(f"**{custom_response['predicted_food']}**")
                                
                                st.subheader("Nutrition Information (per serving):")
                                # Create a nice table for nutrition info
                                nutr_data = []
                                for nutrient, value in custom_response["nutrition"].items():
                                    nutr_data.append({"Nutrient": nutrient, "Amount": value})
                                
                                st.table(nutr_data)
                
                progress_bar.progress(100)
                
            status_text.success("Analysis complete!")
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Run the main function
if __name__ == "__main__":
    main()