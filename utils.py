def format_jobs(jobs: list):
    """Format job list into a display string."""
    return "\n".join([f"{j['title']} - {j['company']} ({j['location']})" for j in jobs])
