# WhatsApp Chat Analysis

## Problem
Analyze WhatsApp chat data to extract meaningful insights about user activity,
message patterns, and communication behavior.

The project focuses on understanding chat statistics rather than prediction.

## Dataset
Exported WhatsApp chat file (`.txt` format).
- Contains timestamps, sender names, and messages
- System messages are handled separately

## Approach
This project applies data analysis and basic NLP techniques to WhatsApp chats.

Workflow:
1. Load WhatsApp chat text file
2. Parse messages using regular expressions
3. Extract date, time, user, and message content
4. Clean and preprocess message text
5. Perform exploratory data analysis (EDA)
6. Visualize chat activity and patterns

## Analysis Performed
- Total messages and words
- Message count per user
- Most active users
- Timeline analysis (daily / monthly activity)
- Word frequency analysis
- Emoji usage analysis (if present)

## Techniques Used
- Regular expressions for text parsing
- Text preprocessing
- Group-by analysis with Pandas
- Data visualization

## Output
- Statistical summaries of chat activity
- Plots showing message trends and user participation
- Insights into communication behavior

## Files
- `prj15Whatzapp Chat Analysis.ipynb` — complete notebook implementation

## Tools & Libraries
- Python
- Pandas, NumPy
- re (Regular Expressions)
- Matplotlib
- Seaborn
- Emoji (if used)

## How to Run
1. Export a WhatsApp chat as a `.txt` file
2. Open the notebook:
   ```bash
   prj15Whatzapp Chat Analysis.ipynb
