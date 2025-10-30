# Flight Stat

A Python CLI tool and library for fetching and displaying Buddha Air flight status information. Features static HTML page generation with historical data filtering.

## Features

- Fetch flight status from Buddha Air API
- Store flight data in SQLite database
- Generate static HTML pages with Bootstrap styling
- Filter flights by date, departure, and arrival airports
- View historical flight data for individual routes
- Interactive web interface with date picker and route filters

## Installation

Install dependencies using `uv`:

```bash
uv sync
```

This will install the package in editable mode along with all dependencies.

## Usage

### CLI Commands

**Update flight data:**
```bash
flight-stat update <FROM> <TO>    # Fetch flights for a specific route
flight-stat update --all           # Fetch flights for all routes
```

**Show flights from database:**
```bash
flight-stat show                  # Show all flights
flight-stat show <FROM> <TO>      # Show flights for a specific route
```

**Generate static pages:**
```bash
flight-stat generate              # Generate HTML pages in output/
flight-stat generate --output-dir dist  # Custom output directory
```

**Serve generated pages:**
```bash
flight-stat serve                # Serve on default port 8000
flight-stat serve --port 8080    # Serve on custom port
```

### Library Usage

```python
from flight_stat import init_db, fetch_flight_status, parse_xml, save_flights

# Initialize database
conn = init_db()

# Fetch and parse flight status
xml_content = fetch_flight_status("KTM", "BIR")
flights = parse_xml(xml_content)

# Save to database
save_flights(conn, flights, "KTM", "BIR")
```

## Development

Format and check code with `ruff`:

```bash
uv run ruff format .
uv run ruff check .
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

