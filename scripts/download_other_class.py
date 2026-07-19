import requests
import os
import time
import csv

from io import BytesIO
from PIL import Image


INAT_OBS = "https://api.inaturalist.org/v1/observations"

HEADERS = {
    "User-Agent": "himalayan-wildlife-classifier/1.0"
}

MIN_SIDE_PX = 200


# Verified iNaturalist taxon IDs
SPECIES = [
    (118552, 100, "domestic_cat"),
    (47144, 100, "dog"),

    (41963, 50, "leopard"),
    (41979, 50, "eurasian_lynx"),
    (42042, 50, "caracal"),

    (41644, 50, "polar_bear"),
    (41638, 50, "american_black_bear"),

    (123070, 50, "domestic_goat"),
    (42351, 50, "siberian_ibex"),
    (121578, 50, "domestic_sheep"),

    (42048, 50, "wolf"),
    (42069, 50, "red_fox"),
]


FIELDNAMES = [
    "filename",
    "filepath",
    "taxon_id",
    "folder",
    "observation_id",
    "observation_url",
    "photo_id",
    "photo_url",
    "license",
    "quality_grade",
    "observed_on",
    "place_guess",
]


def fetch_species(
    taxon_id,
    target_count,
    folder_name,
    out_dir,
    writer,
    max_pages=5
):

    os.makedirs(out_dir, exist_ok=True)

    existing = len([
        f for f in os.listdir(out_dir)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    if existing >= target_count:
        print(
            f"{folder_name}: already have "
            f"{existing}/{target_count}"
        )
        return


    saved = existing
    page = 1


    while saved < target_count and page <= max_pages:

        resp = requests.get(
            INAT_OBS,
            params={
                "taxon_id": taxon_id,
                "photos": "true",
                "quality_grade": "research",
                "license": "cc0,cc-by,cc-by-nc",
                "per_page": 200,
                "page": page,
                "order": "desc",
                "order_by": "created_at",
            },
            headers=HEADERS
        )

        resp.raise_for_status()

        results = resp.json()["results"]

        if not results:
            break


        for obs in results:

            if saved >= target_count:
                break


            obs_taxon = obs.get("taxon") or {}

            # extra safety check
            if obs_taxon.get("id") != taxon_id:
                continue


            if obs.get("quality_grade") != "research":
                continue


            photos = obs.get("photos", [])

            if not photos:
                continue


            photo = photos[0]


            url = photo["url"].replace(
                "square",
                "medium"
            )


            try:
                img_resp = requests.get(
                    url,
                    timeout=10,
                    headers=HEADERS
                )

                img_resp.raise_for_status()


                check = Image.open(
                    BytesIO(img_resp.content)
                )

                check.verify()


                reopened = Image.open(
                    BytesIO(img_resp.content)
                )


                if min(reopened.size) < MIN_SIDE_PX:
                    continue


            except Exception:
                continue



            filename = (
                f"{folder_name}_"
                f"{obs['id']}_"
                f"{photo['id']}.jpg"
            )


            filepath = os.path.join(
                out_dir,
                filename
            )


            if os.path.exists(filepath):
                continue


            with open(filepath, "wb") as f:
                f.write(img_resp.content)



            writer.writerow({

                "filename": filename,

                "filepath": filepath,

                "taxon_id": taxon_id,

                "folder": folder_name,

                "observation_id": obs["id"],

                "observation_url":
                    f"https://www.inaturalist.org/observations/{obs['id']}",

                "photo_id": photo["id"],

                "photo_url": url,

                "license":
                    photo.get("license_code"),

                "quality_grade":
                    obs.get("quality_grade"),

                "observed_on":
                    obs.get("observed_on"),

                "place_guess":
                    obs.get("place_guess"),
            })


            saved += 1



        page += 1
        time.sleep(1)



    print(
        f"{folder_name}: saved "
        f"{saved}/{target_count}"
    )



def main():

    output_dir = (
        "data/raw/gb_wildlife_dataset/other"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    metadata_path = (
        f"{output_dir}/metadata.csv"
    )


    file_exists = os.path.exists(
        metadata_path
    )


    with open(
        metadata_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )


        if not file_exists:
            writer.writeheader()



        for taxon_id, count, folder in SPECIES:

            fetch_species(
                taxon_id,
                count,
                folder,
                f"{output_dir}/{folder}",
                writer
            )



if __name__ == "__main__":
    main()