# US Neighborhood Climate Resilience Analyzer

## Overview
A professional dark-themed climate analysis dashboard for US neighborhoods. Features a three-column layout with a Climate Toolkit sidebar, interactive Mapbox map, and AI Climate Analysis panel. Uses real climate data from Open-Meteo APIs.

## Recent Changes
- 2026-02-14: Added Mapbox GL JS map with dark-v11 style, Mapbox Geocoding for real location search, Open-Meteo APIs for real climate data (temperature, elevation, solar radiation), calculated metrics (flood risk, solar rating, climate zone, energy savings, resilience score), info overlay on map, caching (1hr TTL)
- 2026-02-14: Initial build of the dashboard UI with dark theme, metric cards, intervention cards, sliders, accordion analysis sections

## Project Architecture
- **Frontend**: React + Vite + TypeScript + Tailwind CSS + shadcn/ui + react-map-gl + axios
- **Backend**: Express.js with API proxy routes
- **Map**: react-map-gl/mapbox with mapbox-gl, dark-v11 style
- **Climate Data**: Open-Meteo API (free, no auth) for weather, elevation, solar radiation
- **Geocoding**: Mapbox Geocoding API via backend proxy
- **Routing**: wouter for client-side routing
- **State**: React local state (useState) + React Query for API data
- **Theme**: Always dark mode (forced via document.documentElement.classList)

## Key Files
- `client/src/pages/dashboard.tsx` - Main dashboard page with map, search, climate data, all three columns
- `client/src/App.tsx` - App entry with dark mode and routing
- `client/src/index.css` - Theme variables (cyan primary #06b6d4)
- `server/routes.ts` - Express API routes: /api/mapbox/token, /api/mapbox/geocode, /api/climate/data

## API Endpoints
- `GET /api/mapbox/token` - Returns Mapbox access token from env
- `GET /api/mapbox/geocode?q={query}` - Proxies Mapbox Geocoding API (US places/neighborhoods)
- `GET /api/climate/data?lat={lat}&lon={lon}` - Proxies Open-Meteo APIs (weather, elevation, solar)

## Environment Variables
- `MAPBOX_ACCESS_TOKEN` - Required for map and geocoding functionality

## User Preferences
- Dark theme: background #0f172a, cards #1e293b
- Cyan accents (#06b6d4) throughout
- Inter font family
- Professional climate analysis tool aesthetic
- Three-column layout: 20% / 55% / 25%
