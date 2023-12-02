transcript_summary_template_old = """You create summaries of transcripts of videos.
Given this transcript below, create a summary
Transcript
```
{text}
```
"""

transcript_summary_template = """You create summaries of transcripts of videos.
Given this transcript below, create a summary.

If its a cooking related video, try to create a summary like below with separate headings
 - "Ingredients " :
- "Cooking steps" :
Also mention if its an easy / medium or hard recipe.

If its a learning video, create a summary to include a separate heading for "Key Learnings".

If its a movie video, try to create a summary with some kind of plot

If its a fitness / exercise video, try to create a summary which includes headings for "Key exercises"  and the "Exercise benefits" as points.

Transcript
```
{text}
```
"""


comment_summary_template = """You create summaries of comments in videos.
Given the comments below, summarize them in the following manner:

Overall summary of comments:
What did people like:
What did people not like:

Comments
```
{text}
```
"""
transcript_query_format = """
The context is an extract from a video transcript. It shows the time in brackets
Also return the time in brackets which refers to this answer. Your output format should be like this.Include the time before the answer which refers to this answer.
[{{
    "time":<time>,
    "Answer":<your answer>
}}]
]
For example:
Question:
What did Julian do differently.
Response:
[{{
    "time":"59",
    "Answer":"Julian did her training session differently the day before her performance."
}},
{{
    "time":"66-106"
    "Answer":"She did not follow the common lifting routine of doing maximal snatch and clean and jerk. Instead, she focused on specific warm-up exercises, total body mobility exercises, and different drills with rubber bands."
}},
]
Transcript information is below.
--------------------
{context_str}
---------------------
Query: {query_str}
Answer:
"""

intent_prompt_template = """
A user asks a question about a Youtube video. We have access to the transcript and comments.
Based on the question you need to decide which tool should be called.

If a question relates to video content itself and is at a high level , then choose Generate_Answer_from_Video_Summary.

If a question relates to video content itself and is a specific question which requires an answer from the video, then choose Generate_Answer_from_Video_Detailed.

If a question relates to what people's comments were on the video Generate_Answer_from_Audience_Comments_Detailed.

If a question relates to what people's comments were on the video and is at a higher level, then use Generate_Answer_from_Audience_Comments_Summary.

The tools available to you are :
1.  Generate_Answer_from_Video_Detailed
2. Generate_Answer_from_Audience_Comments_Detailed
3.  Generate_Answer_from_Video_Summary
4. Generate_Answer_from_Audience_Comments_Summary

Your output should be one of these :
"Generate_Answer_from_Video_Detailed" / "Generate_Answer_from_Audience_Comments_Detailed" / "Get_Answer_from_Video_Summary"/ "Generate_Answer_from_Audience_Comments_Summary" / "NONE"

Question:  {query}
"""

comment_summary_qa_template = """
Answer the question given this summary of comments for the video.

```
{summary}
```
Question:  {query}
"""

transcript_summary_qa_template = """
Answer the question given this summary from video transcript

```
{summary}
```
Question:  {query}
"""
