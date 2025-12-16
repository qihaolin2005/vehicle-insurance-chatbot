from fastapi import FastAPI
from pydantic import BaseModel
from uuid import uuid4
import os
import openai
from dotenv import load_dotenv
import httpx
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



load_dotenv()
app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")
ZENQUOTES_API_URL = os.getenv("ZENQUOTES_API_URL")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    conversation_id: str
    response: str

conversations = {}
ask = [
    "Zipcode",
    "Full Name",
    "email in the form 'Please provide your email address:'"
]
vehicle_questions = [
    "Vin in the form 'Please provide your VIN or enter N/A if there it's not available'",
    "Year,Make, and Body Type",
    "Vehicle Use (commuting, commercial, farming, business)",
    "Blind spot warning equipped (Yes or No)",
    "If the vehicle is used for commuting (Days per week used, One way or Round Trip, miles to work/school)",
    "Annual mileage",
    "US License Type (Foreign, Personal, Commercial)",
    "License Status (valid or suspended)"
]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if payload.conversation_id is None:
        conversation_id = str(uuid4())
        conversations[conversation_id] = {
            "messages": [],
            "step": 0,
            "vehicle_step": 0,
            "vehicles": [],
            "current_vehicle": {}
        }
    else:
        conversation_id = payload.conversation_id
        conversations.setdefault(conversation_id, {
            "messages": [],
            "step": 0,
            "vehicle_step": 0,
            "vehicles": [],
            "current_vehicle": {}
        })

    convo = conversations[conversation_id]

    convo["messages"].append({"role": "user", "content": payload.message})
    mood =  await is_annoyed_or_frustrated(payload.message)
    if mood:
        bot_response = mood

    elif convo["step"] < len(ask):
        prompt = ask[convo['step']]
        bot_response = await get_bot_response(prompt)
        convo["step"] += 1

    elif convo["step"] == len(ask):
        v_step = convo["vehicle_step"]

        if "asking_add_vehicle" not in convo:
            convo["asking_add_vehicle"] = False

        if v_step == 0 and not convo["asking_add_vehicle"]:
            if "vin_asked" not in convo:
                bot_response = await get_bot_response(vehicle_questions[v_step])
                convo["vin_asked"] = True  # mark that we asked for VIN
            else:
                vin_input = payload.message.strip()

                if vin_input.lower() == "n/a":
                    bot_response = "No VIN provided. Let's continue with vehicle details."
                    convo["vehicle_step"] += 1
                    del convo["vin_asked"]
                else:
                    is_valid = await validate_vin(vin_input.upper())
                    if is_valid:
                        convo["current_vehicle"]["vin"] = vin_input.upper()
                        bot_response = "VIN is valid! Let's continue with vehicle details."
                        convo["vehicle_step"] += 1
                        del convo["vin_asked"]
                    else:
                        bot_response = "Invalid VIN. Please enter a correct 17-character VIN or type N/A."

        elif 1 <= v_step < len(vehicle_questions) and not convo["asking_add_vehicle"]:
            question_text = vehicle_questions[v_step]
            bot_response = await get_bot_response(question_text)

            key = f"q{v_step}"
            convo["current_vehicle"][key] = payload.message

            convo["vehicle_step"] += 1

        elif v_step == len(vehicle_questions) and not convo["asking_add_vehicle"]:
            convo["vehicles"].append(convo["current_vehicle"])
            convo["current_vehicle"] = {}
            convo["vehicle_step"] = 0
            convo["asking_add_vehicle"] = True
            bot_response = await get_bot_response("Do you want to add another vehicle? (Yes/No)")

        elif convo["asking_add_vehicle"]:
            answer = payload.message.strip().lower()
            if answer in ["yes", "y"]:
                bot_response = vehicle_questions[0]  # start next vehicle
                convo["vehicle_step"] = 0
                convo["asking_add_vehicle"] = False
            else:
                convo["asking_add_vehicle"] = False
                convo["step"] += 1
                bot_response = "All questions completed. Thank you!"

        else:
            bot_response = "All questions completed. Thank you!"

    convo["messages"].append({"role": "assistant", "content": bot_response})

    return ChatResponse(
        conversation_id=conversation_id,
        response=bot_response
    )




async def get_bot_response(question: str) -> str:
    response = openai.responses.create(
        model="gpt-4",
        input = "Ask for a user's" + question,
        temperature=0.7
    )

    return response.output[0].content[0].text

async def validate_vin(vin: str) -> bool:
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{vin}?format=json"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        data = resp.json()
    # If Make is returned, VIN is valid
    make = data.get("Results", [{}])[0].get("Make", "")
    print(make)
    return bool(make)

async def is_annoyed_or_frustrated(text, threshold=-0.4):
    analyzer = SentimentIntensityAnalyzer()

    scores = analyzer.polarity_scores(text)
    if scores["compound"] <= threshold:
        async with httpx.AsyncClient() as client:
            resp = await client.get(ZENQUOTES_API_URL)
            data = resp.json()

            # ZenQuotes returns a list with one item
            quote = data[0]["q"]
            author = data[0]["a"]
            return f"{quote} — {author}"
    else:
        return None




