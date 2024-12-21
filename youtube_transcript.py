from youtube_transcript_api import YouTubeTranscriptApi

# Fetch the transcript
transcript = YouTubeTranscriptApi.get_transcript("3XbtEX3jUv4", languages=['ko'])

# Combine the text fields into a single string
combined_text = " ".join([item['text'] for item in transcript])

# Remove unnecessary extra spaces and newlines
formatted_text = " ".join(combined_text.split())

# Print the formatted text
print(formatted_text)
