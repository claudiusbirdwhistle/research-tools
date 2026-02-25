"""End-to-end test: fetch 3 cities x 5 years, verify data and caching."""

import sys
import logging
import json

sys.path.insert(0, "/tools/climate-trends")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from climate.cache import ResponseCache
from climate.cities import City, get_cities, get_city_batches, get_continents, get_climate_zones
from climate.client import OpenMeteoClient

def test_cities():
    """Verify city definitions."""
    cities = get_cities()
    assert len(cities) == 52, f"Expected 52 cities, got {len(cities)}"
    
    continents = get_continents()
    print(f"Continents ({len(continents)}): {continents}")
    assert len(continents) == 6, f"Expected 6 continents, got {len(continents)}"
    
    zones = get_climate_zones()
    print(f"Climate zones ({len(zones)}): {zones}")
    
    batches = get_city_batches(10)
    print(f"Batches of 10: {len(batches)} batches, sizes: {[len(b) for b in batches]}")
    assert len(batches) == 6  # 52 / 10 = 5 full + 1 partial
    
    print("✓ City definitions OK\n")

def test_historical_fetch():
    """Fetch 3 cities x 5 years of historical data."""
    test_cities_list = [
        City("London", "UK", "Europe", 51.51, -0.13, "Oceanic", 9.0),
        City("Tokyo", "Japan", "Asia", 35.68, 139.69, "Humid subtropical", 37.4),
        City("New York", "USA", "North America", 40.71, -74.01, "Humid subtropical", 18.8),
    ]
    
    cache = ResponseCache(ttl=7 * 86400)
    client = OpenMeteoClient(cache=cache, historical_delay=1.0)
    
    try:
        # First fetch (should hit API)
        print("Fetching 3 cities x 5 years (2019-2023)...")
        result = client.fetch_historical_batch(
            cities=test_cities_list,
            start_date="2019-01-01",
            end_date="2023-12-31",
        )
        
        print(f"  Fetch took {result.elapsed_seconds:.1f}s, from_cache={result.from_cache}")
        print(f"  Got {len(result.data)} city result(s)")
        
        for i, (city, data) in enumerate(zip(result.cities, result.data)):
            daily = data.get("daily", {})
            times = daily.get("time", [])
            t_max = daily.get("temperature_2m_max", [])
            t_min = daily.get("temperature_2m_min", [])
            t_mean = daily.get("temperature_2m_mean", [])
            precip = daily.get("precipitation_sum", [])
            
            # Check we got all 4 variables
            assert "time" in daily, f"Missing 'time' for {city.name}"
            assert "temperature_2m_max" in daily, f"Missing t_max for {city.name}"
            assert "temperature_2m_min" in daily, f"Missing t_min for {city.name}"
            assert "temperature_2m_mean" in daily, f"Missing t_mean for {city.name}"
            assert "precipitation_sum" in daily, f"Missing precip for {city.name}"
            
            # Check date range (5 years = ~1826 days)
            assert len(times) > 1800, f"Expected >1800 days, got {len(times)} for {city.name}"
            assert times[0] == "2019-01-01", f"First date: {times[0]}"
            assert times[-1] == "2023-12-31", f"Last date: {times[-1]}"
            
            # Check array lengths match
            assert len(t_max) == len(times), f"t_max length mismatch for {city.name}"
            assert len(t_min) == len(times), f"t_min length mismatch for {city.name}"
            assert len(t_mean) == len(times), f"t_mean length mismatch for {city.name}"
            assert len(precip) == len(times), f"precip length mismatch for {city.name}"
            
            # Count nulls
            null_count = sum(1 for v in t_mean if v is None)
            null_pct = null_count / len(t_mean) * 100
            
            # Sanity: London summer should have temps > 15°C
            # Just check non-null values exist and are reasonable
            non_null_temps = [v for v in t_mean if v is not None]
            avg_temp = sum(non_null_temps) / len(non_null_temps)
            
            print(f"  {city.name}: {len(times)} days, avg_temp={avg_temp:.1f}°C, "
                  f"nulls={null_count} ({null_pct:.1f}%)")
            
            # Returned lat/lon should be close to requested
            ret_lat = data.get("latitude", 0)
            ret_lon = data.get("longitude", 0)
            lat_diff = abs(ret_lat - city.lat)
            lon_diff = abs(ret_lon - city.lon)
            print(f"    Returned coords: ({ret_lat}, {ret_lon}), "
                  f"diff: ({lat_diff:.2f}°, {lon_diff:.2f}°)")
        
        # Test caching: second fetch should be instant
        print("\nTesting cache (re-fetching same data)...")
        result2 = client.fetch_historical_batch(
            cities=test_cities_list,
            start_date="2019-01-01",
            end_date="2023-12-31",
        )
        # The from_cache check in our FetchResult uses a pre-check so it should be True
        # But let's check the stats instead
        stats = client.stats()
        print(f"  Client stats: {json.dumps(stats, indent=2)}")
        assert stats["cache_hits"] >= 1, "Expected at least 1 cache hit"
        assert result2.elapsed_seconds < 1.0, f"Cache fetch took {result2.elapsed_seconds:.1f}s (should be <1s)"
        
        print("\n✓ Historical fetch + caching OK")
        
    finally:
        client.close()

if __name__ == "__main__":
    test_cities()
    test_historical_fetch()
    print("\n=== ALL TESTS PASSED ===")
