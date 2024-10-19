from langchain_core.tools import tool
import csv

def read_property_tax(property_type: str, property_value: float, constraint: str) -> list[tuple[float, float]]:
    property_taxes = []
    with open('tax.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row, property_type, constraint, property_value)
            if row[0] == property_type and row[1] == constraint:
                property_taxes.append((float(row[2]), float(row[3])))
            
    return property_taxes


@tool
def calculate_tax(property_type: str, constraint: str, value: float) -> float :
    """Calculates_tax based on property type and value.
    property_type: str is "agricultural_land", "house_land", "beneficial_land", "abandoned_land"
    constraints: str, the constraints of the agricultural_land property is normal_person or juristic_person.
    constraints: str, the constraints of the house_land property is primary_residence_with_own_house_and_land, primary_residence_with_own_house_only, secondary_residence.
    in other cases, constraints is empty string.
    value: int, the value of the property in million baht.
    return: float, the tax value in baht.
    """
    # read_property_tax()

 # Define tax brackets with lower bounds (lower limit, tax rate)
    tax_brackets = read_property_tax(property_type, value, constraint)    
    total_tax = 0.0
    
    for i in range(len(tax_brackets)):
        lower_bound, rate = tax_brackets[i]
        if value > lower_bound:
            # Determine the upper bound of the current bracket
            if i + 1 < len(tax_brackets):
                upper_bound = tax_brackets[i + 1][0]
            else:
                upper_bound = float('inf')  # No upper bound for the last bracket
            
            # Calculate the taxable amount in this bracket
            taxable_income = min(value, upper_bound) - lower_bound
            tax = taxable_income * rate / 100
            total_tax += tax
            
            print(f"Taxed {taxable_income} at {rate*100}%: {tax}")
    
    return total_tax * 1e6  # Convert from million baht to baht

def calculate_deprication(property_type, value, years_of_use):
    """Calculates the depreciated value of a property."""
    depreciation_rate = 0
    match property_type:
        case "agricultural_land":
            depreciation_rate = 0.01
        case "house_land":
            depreciation_rate = 0.02
        case _:
            depreciation_rate = 0.015

    depreciated_value = value * ((1 - depreciation_rate) ** years_of_use)
    return depreciated_value, depreciation_rate
