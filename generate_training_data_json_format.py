import pandas as pd
import json
import random
from datetime import datetime

# ---- Configuration ----
input_file = "/home/ayman/LLM_Project/Data_Preparation/dummy_data.xlsx"
output_file_train = "/home/ayman/LLM_Project/Data_Preparation/fine_tuning_train.jsonl"
output_file_eval = "/home/ayman/LLM_Project/Data_Preparation/fine_tuning_eval.jsonl"

schema_description = """Database Schema:
Table: patents
- id (INT): Patent ID
- title (TEXT): Title of the patent
- filing_date (DATE): Date of application
- grant_date (DATE): Date granted
- status (TEXT): Legal status
- assignee (TEXT): Organization owning the patent
- inventors (TEXT): Names of inventors
- ipc (TEXT): International Patent Classification
- abstract (TEXT): Summary of the invention"""

FIELDS = ["id", "title", "filing_date", "grant_date", "status", "assignee", "inventors", "ipc", "abstract"]
TABLE_NAME = "patents"
KEYWORDS = ["wireless", "AI", "battery", "sensor", "robotics", "neural", "solar", "blockchain"]

# ---- Helper functions ----
def to_date_str(val):
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    return str(val)

def clean_field(val):
    if pd.isna(val):
        return ""
    if isinstance(val, str):
        return val.replace("'", "''").strip()
    return str(val).strip()

def random_instruction(template_list):
    return random.choice(template_list)

def add_query(instruction, sql, query_type, explanation=None):
    entry = {
        "instruction": instruction,
        "input": schema_description,
        "output": sql,
        "metadata": {"query_type": query_type}
    }
    if explanation:
        entry["metadata"]["explanation"] = explanation
    return entry

# ---- Query Generator ----
def generate_variations(row):
    queries = []

    row_id = clean_field(row.get("id (Primary)"))
    assignee = clean_field(row.get("applicants"))
    inventor = clean_field(row.get("inventors"))
    grant_date = row.get("grant_date")
    filing_date = row.get("filing_date")
    ipc = clean_field(row.get("ipc"))
    title = clean_field(row.get("title"))

    # 1. Query by ID
    if row_id:
        queries.append(add_query(
            instruction=random_instruction([
                f"Get all info for patent ID {row_id}",
                f"Give me complete data of patent number {row_id}",
                f"Show details for the patent with ID = {row_id}",
            ]),
            sql=f"SELECT {', '.join(FIELDS)} FROM {TABLE_NAME} WHERE id = {row_id};",
            query_type="by_id",
            explanation="Retrieves all fields for a specific patent by ID."
        ))

    # 2. Grant date queries
    if pd.notna(grant_date):
        date_str = to_date_str(grant_date)
        queries.append(add_query(
            instruction=random_instruction([
                f"List patents granted after {date_str}",
                f"Which patents were approved beyond {date_str}?",
                f"Patents with grant date after {date_str}?",
            ]),
            sql=f"SELECT {', '.join(FIELDS)} FROM {TABLE_NAME} WHERE grant_date > '{date_str}';",
            query_type="grant_date"
        ))

    # 3. Filing date range queries
    if pd.notna(filing_date):
        year = pd.to_datetime(filing_date).year
        if year > 2000:
            queries.append(add_query(
                instruction=f"Find patents filed between {year - 2} and {year + 2}",
                sql=f"SELECT {', '.join(FIELDS)} FROM {TABLE_NAME} WHERE filing_date BETWEEN '{year - 2}-01-01' AND '{year + 2}-12-31';",
                query_type="filing_range"
            ))

    # 4. Assignee queries
    if assignee:
        queries += [
            add_query(
                instruction=f"List patents by {assignee}",
                sql=f"SELECT {', '.join(FIELDS)} FROM {TABLE_NAME} WHERE assignee = '{assignee}';",
                query_type="by_assignee"
            ),
            add_query(
                instruction=f"How many patents did {assignee} file?",
                sql=f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE assignee = '{assignee}';",
                query_type="count_by_assignee"
            )
        ]

    # 5. Inventor queries
    if inventor:
        name_sample = inventor.split(",")[0].strip()
        queries.append(add_query(
            instruction=f"Find patents where {name_sample} is an inventor",
            sql=f"SELECT {', '.join(FIELDS)} FROM {TABLE_NAME} WHERE inventors LIKE '%{name_sample}%';",
            query_type="by_inventor"
        ))

    # 6. IPC code
    if ipc:
        ipc_code = ipc.split(" ")[0]
        queries.append(add_query(
            instruction=f"Get all patents under IPC {ipc_code}",
            sql=f"SELECT {', '.join(FIELDS)} FROM {TABLE_NAME} WHERE ipc LIKE '{ipc_code}%';",
            query_type="by_ipc"
        ))

    # 7. Keyword query
    keyword = random.choice(KEYWORDS)
    queries.append(add_query(
        instruction=f"Find all patents related to {keyword}",
        sql=f"SELECT {', '.join(FIELDS)} FROM {TABLE_NAME} WHERE title LIKE '%{keyword}%';",
        query_type="by_keyword"
    ))

    # 8. Contrastive example (invalid column)
    queries.append(add_query(
        instruction="List patents with expiry_date before 2020",
        sql="ERROR: Column 'expiry_date' does not exist in table 'patents'.",
        query_type="negative_example"
    ))

    # 9. Irrelevant question (no-op)
    queries.append(add_query(
        instruction="Who won the world cup in 2018?",
        sql="ERROR: Instruction unrelated to the database schema.",
        query_type="irrelevant"
    ))

    return queries

# ---- Load Excel and Generate ----
df = pd.read_excel(input_file)
data = []
for _, row in df.iterrows():
    data.extend(generate_variations(row))

# ---- Shuffle and Split ----
random.seed(42)
random.shuffle(data)
split_index = int(0.85 * len(data))
train_set, eval_set = data[:split_index], data[split_index:]

# ---- Save as JSONL ----
def save_jsonl(filename, records):
    with open(filename, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(output_file_train, train_set)
save_jsonl(output_file_eval, eval_set)

print(f"✅ Generated {len(train_set)} training and {len(eval_set)} evaluation examples.")
