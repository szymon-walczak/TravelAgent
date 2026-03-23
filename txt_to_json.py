import re
import json

def mpg_to_l_per_100km(mpg):
    if not mpg or mpg == 0: return 0
    # Standard formula for conversion
    return round(235.215 / mpg, 2)

def preprocess_feg_txt(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        # Filter source tags and empty lines immediately
        lines = [line.strip() for line in f if line.strip() and not line.startswith('[source')]

    categories_set = {
        "TWO-SEATER CARS", "MINICOMPACT CARS", "SUBCOMPACT CARS",
        "COMPACT CARS", "MIDSIZE CARS", "LARGE CARS",
        "SMALL STATION WAGONS", "MIDSIZE STATION WAGONS",
        "SMALL PICKUP TRUCKS 2WD", "SMALL PICKUP TRUCKS 4WD",
        "STANDARD PICKUP TRUCKS 2WD", "STANDARD PICKUP TRUCKS 4WD",
        "MINIVANS 2WD", "MINIVANS 4WD",
        "SMALL SPORT UTILITY VEHICLES 2WD", "SMALL SPORT UTILITY VEHICLES 4WD",
        "STANDARD SPORT UTILITY VEHICLES 2WD", "STANDARD SPORT UTILITY VEHICLES 4WD"
    }

    mfr_whitelist = {
        "ACURA", "ALFA ROMEO", "ASTON MARTIN", "AUDI", "BENTLEY", "BMW", "BUGATTI",
        "BUGATTI RIMAC", "BUICK", "CADILLAC", "CHEVROLET", "CHRYSLER", "DODGE",
        "FERRARI", "FIAT", "FORD", "GENESIS", "GMC", "HONDA", "HYUNDAI", "INEOS",
        "INFINITI", "JAGUAR", "JEEP", "KIA", "LAMBORGHINI", "LAND ROVER", "LEXUS",
        "LINCOLN", "LOTUS", "LUCID", "MASERATI", "MAZDA", "MERCEDES-BENZ", "MINI",
        "MITSUBISHI", "NISSAN", "POLESTAR", "PORSCHE", "RAM", "RIVIAN", "ROLLS-ROYCE",
        "SUBARU", "TESLA", "TOYOTA", "VINFAST", "VOLKSWAGEN", "VOLVO"
    }

    structured_data = []
    current_category = "Unknown"
    current_mfr = "Unknown"
    pending_model = ""

    # State tracking for vertical spec data
    pending_specs = None
    pending_comb = None
    pending_city_hwy = None

    # Vertical Patterns [cite: 104, 203, 364]
    config_pattern = re.compile(r'^(?P<trans>[A-Z0-9\-]+),\s*(?P<eng>\d+\.\d+)L,\s*(?P<cyl>\d+)cyl')
    mpg_pattern = re.compile(r'^\d+$')
    city_hwy_pattern = re.compile(r'^(?P<city>\d+)/(?P<hwy>\d+)$')
    cost_pattern = re.compile(r'^\$(?P<cost>[\d,]+)$')

    for line in lines:
        # 1. Update Category
        if line.upper() in categories_set:
            current_category = line.upper()
            continue

        # 2. Update Manufacturer
        if line.upper() in mfr_whitelist:
            current_mfr = line.upper()
            continue

        # 3. Match Config line (e.g., AM-6, 4.7L, 8cyl)
        config_match = config_pattern.match(line)
        if config_match:
            pending_specs = config_match.groupdict()
            continue

        # 4. Match Combined MPG line (e.g., 16)
        if pending_specs and mpg_pattern.match(line):
            pending_comb = int(line)
            continue

        # 5. Match City/Hwy line (e.g., 13/21)
        city_hwy_match = city_hwy_pattern.match(line)
        if pending_comb and city_hwy_match:
            pending_city_hwy = city_hwy_match.groupdict()
            continue

        # 6. Match Cost and FINALIZE
        cost_match = cost_pattern.match(line)
        if pending_city_hwy and cost_match:
            structured_data.append({
                "category": current_category,
                "manufacturer": current_mfr,
                "model": pending_model,
                "engine": f"{pending_specs['eng']}L {pending_specs['cyl']}cyl",
                "transmission": pending_specs['trans'],
                "l_100km_combined": mpg_to_l_per_100km(pending_comb),
                "l_100km_city": mpg_to_l_per_100km(int(pending_city_hwy['city'])),
                "l_100km_highway": mpg_to_l_per_100km(int(pending_city_hwy['hwy'])),
                "annual_cost_usd": int(cost_match.group('cost').replace(',', ''))
            })
            # Reset specs state, but KEEP pending_model for the next trim level!
            pending_specs = pending_comb = pending_city_hwy = None
            continue

        # 7. Model detection logic (The FIX)
        # Avoid capturing technical notes OR data that looks like specs/costs
        technical_notes = ["PR", "P ", "T ", "SS", "Tax", "HEV", "EV", "CD", "MHEV", "PHEV", "See page", "FUEL ECONOMY"]
        is_data = config_pattern.match(line) or mpg_pattern.match(line) or city_hwy_pattern.match(line) or cost_pattern.match(line)

        if not is_data and not any(note in line for note in technical_notes):
            clean_name = line.replace("►", "").strip()
            # Only update the model if it's not a single digit and looks like a real name
            if len(clean_name) > 1 and not clean_name.isdigit():
                pending_model = clean_name

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=4)

    return f"Successfully processed {len(structured_data)} vehicles."

if __name__ == "__main__":
    for year in range(2012, 2026):
        print(preprocess_feg_txt(f"debug_txt/FEG{year}.pdf.txt", f"data/vehicles_processed_{year}.json"))