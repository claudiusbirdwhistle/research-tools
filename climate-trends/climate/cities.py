"""City definitions for climate trend analysis.

52 cities across 6 inhabited continents and 10+ climate zones.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    country: str
    continent: str
    lat: float
    lon: float
    climate: str
    pop_millions: float


# fmt: off
CITIES: list[City] = [
    # Europe (10)
    City("London",      "UK",           "Europe",        51.51,   -0.13,  "Oceanic",              9.0),
    City("Paris",       "France",       "Europe",        48.86,    2.35,  "Oceanic/Continental",  11.0),
    City("Moscow",      "Russia",       "Europe",        55.76,   37.62,  "Humid continental",    12.5),
    City("Madrid",      "Spain",        "Europe",        40.42,   -3.70,  "Mediterranean",         6.7),
    City("Berlin",      "Germany",      "Europe",        52.52,   13.41,  "Oceanic/Continental",   3.6),
    City("Rome",        "Italy",        "Europe",        41.90,   12.50,  "Mediterranean",         4.3),
    City("Stockholm",   "Sweden",       "Europe",        59.33,   18.07,  "Humid continental",     1.6),
    City("Athens",      "Greece",       "Europe",        37.98,   23.73,  "Mediterranean",         3.2),
    City("Istanbul",    "Turkey",       "Europe",        41.01,   28.98,  "Mediterranean",        15.6),
    City("Reykjavik",   "Iceland",      "Europe",        64.15,  -21.94,  "Subpolar oceanic",      0.2),

    # Asia (12)
    City("Tokyo",       "Japan",        "Asia",          35.68,  139.69,  "Humid subtropical",    37.4),
    City("Beijing",     "China",        "Asia",          39.91,  116.40,  "Humid continental",    21.5),
    City("New Delhi",   "India",        "Asia",          28.61,   77.21,  "Semi-arid",            32.9),
    City("Mumbai",      "India",        "Asia",          19.08,   72.88,  "Tropical monsoon",     21.7),
    City("Shanghai",    "China",        "Asia",          31.23,  121.47,  "Humid subtropical",    28.5),
    City("Singapore",   "Singapore",    "Asia",           1.35,  103.82,  "Tropical rainforest",   5.9),
    City("Dubai",       "UAE",          "Asia",          25.27,   55.30,  "Arid desert",           3.6),
    City("Bangkok",     "Thailand",     "Asia",          13.76,  100.50,  "Tropical savanna",     10.7),
    City("Seoul",       "South Korea",  "Asia",          37.57,  126.98,  "Humid continental",     9.7),
    City("Jakarta",     "Indonesia",    "Asia",          -6.17,  106.85,  "Tropical rainforest",  34.5),
    City("Riyadh",      "Saudi Arabia", "Asia",          24.63,   46.72,  "Arid desert",           7.7),
    City("Ulaanbaatar", "Mongolia",     "Asia",          47.91,  106.91,  "Continental extreme",   1.5),

    # Africa (8)
    City("Cairo",       "Egypt",        "Africa",        30.04,   31.24,  "Arid desert",          22.2),
    City("Lagos",       "Nigeria",      "Africa",         6.52,    3.38,  "Tropical savanna",     15.9),
    City("Nairobi",     "Kenya",        "Africa",        -1.29,   36.82,  "Subtropical highland",  5.1),
    City("Johannesburg","South Africa", "Africa",       -26.20,   28.05,  "Subtropical highland",  6.1),
    City("Casablanca",  "Morocco",      "Africa",        33.59,   -7.62,  "Mediterranean",         3.8),
    City("Addis Ababa", "Ethiopia",     "Africa",         9.02,   38.75,  "Subtropical highland",  5.5),
    City("Kinshasa",    "DR Congo",     "Africa",        -4.44,   15.27,  "Tropical wet/dry",     17.7),
    City("Dakar",       "Senegal",      "Africa",        14.69,  -17.44,  "Semi-arid",             3.9),

    # North America (8)
    City("New York",    "USA",          "North America", 40.71,  -74.01,  "Humid subtropical",    18.8),
    City("Los Angeles", "USA",          "North America", 34.05, -118.24,  "Mediterranean",        12.5),
    City("Chicago",     "USA",          "North America", 41.88,  -87.63,  "Humid continental",     8.6),
    City("Houston",     "USA",          "North America", 29.76,  -95.37,  "Humid subtropical",     7.1),
    City("Mexico City", "Mexico",       "North America", 19.43,  -99.13,  "Subtropical highland", 21.8),
    City("Toronto",     "Canada",       "North America", 43.65,  -79.38,  "Humid continental",     6.4),
    City("Miami",       "USA",          "North America", 25.76,  -80.19,  "Tropical monsoon",      6.1),
    City("Phoenix",     "USA",          "North America", 33.45, -112.07,  "Arid desert",           4.9),

    # South America (7)
    City("São Paulo",    "Brazil",      "South America",-23.55,  -46.63,  "Humid subtropical",    22.4),
    City("Buenos Aires", "Argentina",   "South America",-34.60,  -58.38,  "Humid subtropical",    15.5),
    City("Bogotá",       "Colombia",    "South America",  4.71,  -74.07,  "Subtropical highland", 11.3),
    City("Lima",         "Peru",        "South America",-12.05,  -77.04,  "Arid coastal",         11.0),
    City("Santiago",     "Chile",       "South America",-33.45,  -70.65,  "Mediterranean",         6.8),
    City("Rio de Janeiro","Brazil",     "South America",-22.91,  -43.17,  "Tropical savanna",     13.6),
    City("Quito",        "Ecuador",     "South America", -0.18,  -78.47,  "Subtropical highland",  2.8),

    # Oceania (4)
    City("Sydney",      "Australia",    "Oceania",      -33.87,  151.21,  "Humid subtropical",     5.4),
    City("Melbourne",   "Australia",    "Oceania",      -37.81,  144.96,  "Oceanic",               5.0),
    City("Auckland",    "New Zealand",  "Oceania",      -36.85,  174.76,  "Oceanic",               1.7),
    City("Perth",       "Australia",    "Oceania",      -31.95,  115.86,  "Mediterranean",         2.1),

    # Extreme/Frontier (3)
    City("Anchorage",   "USA",          "North America", 61.22, -149.90,  "Subarctic",             0.3),
    City("Karachi",     "Pakistan",     "Asia",          24.86,   67.01,  "Arid coastal",         16.8),
    City("Novosibirsk", "Russia",       "Asia",          55.01,   82.93,  "Continental extreme",   1.6),
]
# fmt: on


def get_cities() -> list[City]:
    """Return all 52 cities."""
    return CITIES


def get_cities_by_continent(continent: str) -> list[City]:
    """Return cities for a given continent."""
    return [c for c in CITIES if c.continent == continent]


def get_city_batches(batch_size: int = 10) -> list[list[City]]:
    """Split cities into batches for batch API requests."""
    return [CITIES[i:i + batch_size] for i in range(0, len(CITIES), batch_size)]


def get_continents() -> list[str]:
    """Return sorted list of unique continents."""
    return sorted(set(c.continent for c in CITIES))


def get_climate_zones() -> list[str]:
    """Return sorted list of unique climate zones."""
    return sorted(set(c.climate for c in CITIES))
