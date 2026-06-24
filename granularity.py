# granularity.py
import os
import requests
import pandas as pd
import base64
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI


import os
import json
import base64
import argparse
import pandas as pd

import re

# Load environment variables in a file called .env

load_dotenv(override=True)

api_key = os.getenv(
    "OPENAI_API_KEY"
)

# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")





# =====================================================
# IMAGE
# =====================================================

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")


# =====================================================
# PROMPTS
# =====================================================

SYSTEM_PROMPT = """
You are an expert assistant for affordance-aware Approximate Convex Decomposition (ACD) of interactive 3D game objects.

Your task is to analyze mesh-face affordance annotations and estimate how finely different mesh regions should be decomposed for real-time game physics and collision systems.

The target environment is a real-time game engine.

The goal is NOT perfect geometric reconstruction.
Instead, the goal is to balance:

- gameplay interaction fidelity
- collision precision
- physics stability
- runtime performance
- real-time simulation cost

Do NOT estimate decomposition importance primarily from affordance region size or face count.
Small but interaction-critical regions may require significantly finer decomposition than large passive regions.


Input CSV fields:
- face_id: mesh triangle index
- v1,v2,v3: triangle vertex indices
- affordance: semantic affordance label

You must estimate which affordance regions require:
- fine-grained convex decomposition
- moderate decomposition
- coarse decomposition

Assign an ACD granularity score between 0.0 and 1.0.

Scoring criteria:

0.0 ~ 0.3:
Very coarse decomposition acceptable.
Used for passive or low-interaction regions.

0.3 ~ 0.6:
Moderate decomposition.
General gameplay interaction support.

0.6 ~ 0.8:
Fine decomposition required.
Important gameplay interaction regions.

0.8 ~ 1.0:
Very fine decomposition required.
High-precision interaction or collision-critical regions.

Prioritize fine decomposition for:
- gameplay-critical collision areas
- regions requiring precise player interaction


Prefer coarse decomposition for:
- decorative regions
- passive surfaces
- large static support areas
- visually important but interaction-unimportant geometry

The decomposition strategy should optimize both:
1. interaction realism
2. runtime performance efficiency

Avoid unnecessarily fine decomposition on low-importance regions.

Output ONLY valid JSON.

Example output:

{
  "affordance_regions": [
    {
      "affordance":
      "acd_granularity":
      "priority":
      "reason":
    },
  ]
}
"""


def make_user_prompt(
    csv_text,
    rule_text,
    object_name
):

     user_prompt = f"""
        You are analyzing a 3D game object named "{object_name}".

        Do NOT estimate decomposition importance primarily from affordance region size or face count.
        Small but interaction-critical regions may require significantly finer decomposition than large passive regions.

        List of affordance mapped color:

        affordance_colors = [

            "handle_grasp": [1.0, 0.0, 0.0],   # Red (빨강)

            "none":        [0.8, 0.8, 0.8],   # Light Gray (연회색)

            "grasp":       [1.0, 0.0, 0.0],   # Red (빨강)
            "contain":     [0.0, 1.0, 0.0],   # Green (초록)
            "lift":        [0.0, 0.0, 1.0],   # Blue (파랑)
            "openable":    [1.0, 1.0, 0.0],   # Yellow (노랑)

            "support":     [1.0, 0.5, 0.0],   # Orange (주황)
            "wrap_grasp":  [0.6, 0.0, 1.0],   # Purple (보라)

            "pourable":    [0.0, 1.0, 1.0],   # Cyan (시안 - 하늘)

            "pushable":    [0.4, 0.4, 0.4],   # Gray (회색)
            "pull":        [0.5, 0.25, 0.0],  # Brown (갈색)

            "listen":      [0.0, 0.0, 0.5],   # Navy (남색)

            "wear":        [1.0, 0.4, 0.7],   # Pink (분홍)

            "press":       [0.4, 0.8, 1.0],   # Sky Blue (하늘색)

            "cut":         [0.0, 0.0, 0.0],   # Black (검정)

            "stab":        [1.0, 1.0, 1.0],   # White (흰색)

            "layable":   [1.0, 0.0, 1.0],   # Magenta (자홍)
            "sittable":  [0.0, 0.8, 0.8],   # Teal Cyan (청록)

            "move":      [0.0, 0.6, 1.0],   # Deep Sky Blue (진한 하늘색)
            "display":   [1.0, 0.0, 0.5],   # Hot Pink (진분홍)
        ]

        The following "{rule_text}" contains information about the rule of game.
        So you can discuss which part of the object the player will interact importantly.
        If affordance word is written in the rule explicitly, the affordance is important.

        The following CSV contains mesh-face affordance annotations.

        CSV format:
        - face_id : triangle index
        - v1,v2,v3 : mesh vertex indices
        - affordance : semantic affordance label

        Your task:
        1. Analyze affordance distribution.
        2. Determine which regions require fine-grained ACD.
        3. Determine which regions can use coarse decomposition.
        4. Estimate affordance-aware decomposition importance for real-time game physics and collision.

        The CSV content is below:

        {csv_text}

        Please output:
        - affordance region importance
        - ACD granularity score (0.0 ~ 1.0)
        - decomposition priority
        - reasoning

        Sort the JSON array in descending order of ACD granularity score.

        Respond ONLY in valid JSON.
        """

     return user_prompt


# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True
    )

    parser.add_argument(
        "--image",
        required=False
    )

    parser.add_argument(
        "--rule",
        required=True
    )

    parser.add_argument(
        "--object",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # API KEY
    # ---------------------------------------------

    load_dotenv()

    

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )
    client = OpenAI(
        api_key=api_key
    )

    # ---------------------------------------------
    # LOAD CSV
    # ---------------------------------------------

    df = pd.read_csv(
        args.csv
    )

    summary = (
        df.groupby("affordance")
          .agg(
              face_count=("face_id", "count"),
              min_face_id=("face_id", "min"),
              max_face_id=("face_id", "max")
          )
          .reset_index()
    )

    csv_text = summary.to_csv(
        index=False
    )

    # ---------------------------------------------
    # LOAD RULE
    # ---------------------------------------------

    with open(
        args.rule,
        "r",
        encoding="utf-8"
    ) as f:

        rule_text = f.read()

    # ---------------------------------------------
    # USER PROMPT
    # ---------------------------------------------

    user_prompt = make_user_prompt(
        csv_text,
        rule_text,
        args.object
    )

    # ---------------------------------------------
    # IMAGE
    # ---------------------------------------------

    content = [
        {
            "type":"text",
            "text":user_prompt
        }
    ]

    if args.image:

        base64_image = encode_image(
            args.image
        )

        content.append(
            {
                "type":"image_url",
                "image_url":{
                    "url":
                    f"data:image/jpeg;base64,{base64_image}"
                }
            }
        )

    # ---------------------------------------------
    # GPT
    # ---------------------------------------------

    response = client.chat.completions.create(
        model="gpt-4o-mini",

        messages=[
            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content":content
            }
        ]
    )

    result_text = (
        response
        .choices[0]
        .message
        .content
    )

    

    match = re.search(
        r"\{.*\}",
        result_text,
        re.DOTALL
    )

    result_json = json.loads(
        match.group(0)
    )

    print(result_text)

    # ---------------------------------------------
    # JSON
    # ---------------------------------------------

    # result_json = json.loads(
    #     result_text
    # )

    rows = []

    for item in result_json[
        "affordance_regions"
    ]:

        rows.append(
            {
                "affordance":
                item["affordance"],

                "acd_granularity":
                item["acd_granularity"],

                "priority":
                item["priority"],

                "reason":
                item["reason"]
            }
        )

    out_df = pd.DataFrame(
        rows
    )

        #LLM json -> dataframe
    region_df = pd.DataFrame(
        result_json["affordance_regions"]
    )

    #affordance 기준 merge
    merged_df = df.merge(
        region_df,
        on = "affordance",
        how="left"
    )

    merged_df.to_csv(
        args.output,
        index=False
    )

    print(
        "saved:",
        args.output
    )


if __name__ == "__main__":
    main()