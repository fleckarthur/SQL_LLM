import os
import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# ==== Config ====
output_dir = "LLM_Project/Data_Preparation"
os.makedirs(output_dir, exist_ok=True)

fake = Faker()
num_rows = 5000  # Scaled up to match server capacity

# ==== Date & None Helpers ====
def random_date(start_year=2010, end_year=2023):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    delta = end_date - start_date
    return (start_date + timedelta(days=random.randint(0, delta.days))).isoformat()

def maybe_none(value, probability=0.1):
    return None if random.random() < probability else value

# ==== Data Generation ====
data = []
for i in range(1, num_rows + 1):
    row = {
        "metadata_id": f"patent_{i:06d}",
        "id (Primary)": i,
        "docketidIndex": maybe_none(random.randint(1, 9999)),
        "casenumberIndex": maybe_none(f"CN-{random.randint(10000, 99999)}"),
        "combined_inventors": maybe_none(", ".join(fake.name() for _ in range(random.randint(1, 3)))),
        "inforce": maybe_none(random.randint(0, 1)),
        "priority": maybe_none(random.randint(0, 1)),
        "priority_date": maybe_none(random_date()),
        "due_date_national": maybe_none(random_date()),
        "current_statusIndex": maybe_none(random.randint(1, 100)),
        "current_status_date": maybe_none(random_date()),
        "filing_date": maybe_none(random_date()),
        "is_first_filing": maybe_none(random.randint(0, 1)),
        "due_date_foreign": maybe_none(random_date()),
        "expiry_date": maybe_none(random_date()),
        "grant_date": maybe_none(random_date()),
        "is_first_grant": maybe_none(random.randint(0, 1)),
        "grant_number": maybe_none(f"GN-{random.randint(1000, 9999)}"),
        "title": maybe_none(fake.sentence(nb_words=10)),
        "publication_date": maybe_none(random_date()),
        "publicationno": maybe_none(f"PN-{random.randint(1000, 9999)}"),
        "publicationlink": maybe_none(fake.url()),
        "espace_publink": maybe_none(fake.url()),
        "is_fam_publication": maybe_none(random.randint(0, 1)),
        "memotech_pubno": maybe_none(f"MPN-{random.randint(100, 999)}"),
        "memotech_publink": maybe_none(fake.url()),
        "casekey": maybe_none(fake.lexify(text="???????????")[:11]),
        "archived": maybe_none(random.choice(["Yes", "No"])),
        "first_filing": maybe_none(fake.sentence(nb_words=6)),
        "first_filing_date": maybe_none(random_date()),
        "first_priority": maybe_none(fake.word()),
        "applicants": maybe_none(", ".join(fake.company() for _ in range(random.randint(1, 2)))),
        "legal_owners": maybe_none(", ".join(fake.company() for _ in range(random.randint(1, 2)))),
        "registered_owners": maybe_none(", ".join(fake.company() for _ in range(random.randint(1, 2)))),
        "product_structure_stc": maybe_none(fake.sentence(nb_words=8)),
        "export_restriction": maybe_none(random.choice(["None", "Restricted", "Limited"])),
        "accession_number": maybe_none(f"ACC-{random.randint(1000,9999)}"),
        "case_fc_count": maybe_none(random.randint(0, 50)),
        "countryidIndex": maybe_none(random.randint(1, 250)),
        "filingtypeidIndex": maybe_none(random.randint(1, 10)),
        "filing_number": maybe_none(f"FNUM-{random.randint(1000, 9999)}"),
        "prosecution_status_idIndex": maybe_none(random.randint(1, 100)),
        "sbuidIndex": maybe_none(random.randint(1, 500)),
        "buidIndex": maybe_none(random.randint(1, 500)),
        "blidIndex": maybe_none(random.randint(1, 500)),
        "latest_annuity_number": maybe_none(random.randint(1, 100)),
        "current_annuity_caseIndex": maybe_none(random.randint(1, 100)),
        "current_attorneyIndex": maybe_none(random.randint(1, 100)),
        "current_activated": maybe_none(random.randint(0, 1)),
        "current_activation_date": maybe_none(random_date()),
        "current_decision": maybe_none(random.randint(0, 5)),
        "current_decision_date": maybe_none(random_date()),
        "decision_taken_before_days": maybe_none(random.randint(0, 365)),
        "current_review_status": maybe_none(random.randint(0, 5)),
        "prm_current_review_status": maybe_none(random.randint(0, 5)),
        "filenumberIndex": maybe_none(random.randint(1, 1000)),
        "typename": maybe_none(fake.word().capitalize()),
        "patent_design_number": maybe_none(f"PDN-{random.randint(100,999)}"),
        "addition_timestamp": datetime.now().isoformat(),
        "has_annuity_data": maybe_none(random.randint(0, 1)),
        "application_number": maybe_none(f"APP-{random.randint(1000,9999)}"),
        "pct_filing_date": maybe_none(random_date()),
    }
    data.append(row)

# ==== Save as Excel ====
df = pd.DataFrame(data)
output_filename = os.path.join(output_dir, "dummy_data.xlsx")
df.to_excel(output_filename, index=False)

print(f"✅ Dummy Excel file with {num_rows} rows saved at: {output_filename}")
