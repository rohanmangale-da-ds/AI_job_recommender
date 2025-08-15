from .data_loader import load_job_data
from .embedding_store import JobVectorStore
from .llm_handler import get_llm_response

def recommend(user_input: str, job_file: str = "data/jobs_data.json"):
    jobs_df = load_job_data(job_file)

    vector_store = JobVectorStore()
    vector_store.create_store(jobs_df)

    similar_jobs = vector_store.search(user_input, k=5)

     # Prepare messages for LLM
    user_messages = [
        {"role": "system", "content": "You are a helpful AI that recommends jobs based on user input and similar job listings."},
        {"role": "user", "content": f"User query: {user_input}\nSimilar jobs: {similar_jobs}\nPlease recommend the best matches."}
    ]

    # Get recommendation text from LLM
    llm_response = get_llm_response(user_messages)

    return llm_response, similar_jobs

    # return recommendations, similar_jobs
