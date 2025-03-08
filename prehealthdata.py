import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model_name="llama-3.3-70b-versatile")


class PreHealthData(BaseModel):
    isStressed: bool = Field(description='Is the user stressed?')
    percentStressed: float = Field(description='Percentage of stress')
    stressLevel: str = Field(description='Stress level of the user')
    remedies: str = Field(description='Remedies for the stress level')


parser = PydanticOutputParser(pydantic_object=PreHealthData)


def PreHealthDataOutput(stress):
    template = """You will be given stress percentage and you have to provide:
    isStressed: bool = Field(description='Is the user stressed?') Only give ['True' or 'False']
    percentStressed: float = Field(description='Percentage of stress')
    stressLevel: str = Field(description='Stress level of the user') Only give ['low', 'moderate', 'high']
    remedies: str = Field(description='Remedies for the stress level')
    Give me these three outputs only based on the stress percentage:
    Stress percentage = {stress}

    Stress Level	    Stress Percentage	Recommended Sleep Duration
    low                     50 - 65	                     7-9 hours
    moderate                65 - 85	                    8-9.5 hours
    high                    85 - 100	                8.5-10 hours

    Explanation:
    Below 50% probability → Non-stressed (not considered stressed at all).
    50%\ - 65% → Low Stress (Mild stress, occasional stress triggers).
    65%\ - 85% → Medium Stress (Noticeable stress affecting emotions and productivity).
    85%\ - 100% → High Stress (Severe stress, possible burnout, needs immediate attention).

    Give Remedies based on the stress level.:
    Stress Level	Recommended Remedies	Lifestyle Changes
    Low Stress (50%\ - 65%)	- Deep breathing exercises (4-7-8 technique) 🧘‍♂
    - Light stretching or yoga 🧎‍♂
    - Listening to calming music 🎵
    - Short nature walks 🌿
    - Engaging in hobbies (reading, painting, etc.) 🎨	- Maintain a consistent sleep schedule 🛌
    - Reduce screen time before bed 📵
    - Drink herbal tea like chamomile 🍵
    - Limit caffeine and alcohol intake ☕🚫
    Medium Stress (65%\ - 85%)	- Mindfulness meditation (10-15 mins) 🧘‍♀
    - Regular physical activity (jogging, swimming, cycling) 🏃‍♂
    - Journaling emotions & thoughts 📖
    - Practicing gratitude (writing 3 positive things daily) 🙏
    - Breathing exercises (box breathing technique) 🌬	- Follow a structured daily routine ⏳
    - Reduce workload & multitasking 📉
    - Take short breaks during work/study ☕
    - Eat a balanced diet (rich in omega-3 & magnesium) 🥑🥦
    - Limit exposure to stressful environments 🚶‍♂
    High Stress (85%\ - 100%)	- Guided sleep meditation 💤
    - Intense workouts (HIIT, strength training) 💪
    - Talking to a trusted friend or therapist 🗣
    - Practicing progressive muscle relaxation (PMR) 🏋‍♂
    - Engaging in deep relaxation techniques (hot bath, aromatherapy) 🛀	- Seek professional counseling or therapy 🏥
    - Digital detox (reduce social media & news exposure) 📵
    - Set strict work-life boundaries ⏰
    - Incorporate spiritual or mindfulness practices 🙏
    - Consider stress-management workshops
    {format_instructions}
    """

    prompt = PromptTemplate(template=template,
                            input_variables=["stress"],
                            partial_variables={
                                "format_instructions": parser.get_format_instructions()}
                            )
    chain = prompt | llm | parser

    try:
        response = chain.invoke(
            {"stress": stress, "format_instructions": parser.get_format_instructions()}
        )
        result = {
            "isStressed": response.isStressed,
            "percentStressed": response.percentStressed,
            "stressLevel": response.stressLevel,
            "remedies": response.remedies
        }
        return result
    except Exception as e:
        result = {"error": str(e)}


# def sleep_routine(stress):

#     template = """You will be given stress levels and you have to provide the recommended sleep hours based on the stress level. The stress levels are categorized into Low Stress, Medium Stress, and High Stress. The sleep hours are also categorized into 7-9 hours, 8-9.5 hours, and 8.5-10 hours based on the stress level. The stress levels are categorized as follows:
#     Give me the sleep hours only on the basis of :

#     Stress Level	Probability Threshold (Model Output)	Recommended Sleep Duration
#     Low Stress	50 - 65	7-9 hours
#     Medium Stress	65 - 85	8-9.5 hours
#     High Stress	85 - 100	8.5-10 hours
#     Explanation:
#     Below 50% probability → Non-stressed (not considered stressed at all).
#     50%\ - 65% → Low Stress (Mild stress, occasional stress triggers).
#     65%\ - 85% → Medium Stress (Noticeable stress affecting emotions and productivity).
#     85%\ - 100% → High Stress (Severe stress, possible burnout, needs immediate attention).
#     Stress Level: {stress}
#     """
#     prompt = ChatPromptTemplate(template)
if __name__ == "__main__":
    stress = 70
    print(PreHealthDataOutput(stress))
