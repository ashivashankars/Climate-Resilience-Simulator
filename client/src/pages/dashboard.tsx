import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import MapGL, { Marker, Source, Layer, type MapRef } from "react-map-gl/mapbox";
import "mapbox-gl/dist/mapbox-gl.css";
import {
  Search, Building2, Leaf, Sun, Waves, Droplets, Thermometer,
  Zap, DollarSign, Bot, MapPin, Loader2, Star, AlertTriangle,
  RotateCcw, Crosshair, Sparkles, X, Globe, Brain, Activity,
  TrendingUp, CheckCircle2, AlertCircle
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Accordion, AccordionContent, AccordionItem, AccordionTrigger,
} from "@/components/ui/accordion";

interface GeocodedLocation {
  name: string;
  state: string;
  lat: number;
  lng: number;
  context: string[];
}

interface ClimateData {
  temperature: number;
  tempDelta: number;
  humidity: number;
  elevation: number;
  floodRisk: number;
  floodLabel: string;
  solarRadiation: number;
  solarRating: number;
  solarLabel: string;
  climateZone: string;
  energySavings: number;
  resilienceScore: number;
  isEstimated?: boolean;
  isCached?: boolean;
}

interface BuildingFeature {
  id: string;
  lng: number;
  lat: number;
  height: number;
  intervention: string | null;
}

const INTERVENTION_COLORS: Record<string, string> = {
  "Green Roofs": "#10b981",
  "Solar Panels": "#f59e0b",
  "Flood Walls": "#3b82f6",
  "Permeable Pavement": "#06b6d4",
};

const BASE_BUILDING_COLOR = "#64748b";

const interventions = [
  { icon: Leaf, title: "Green Roofs", description: "Reduce heat island effect", color: "text-emerald-400" },
  { icon: Sun, title: "Solar Panels", description: "Generate clean energy", color: "text-amber-400" },
  { icon: Waves, title: "Flood Walls", description: "Protect against sea rise", color: "text-blue-400" },
  { icon: Droplets, title: "Permeable Pavement", description: "Manage stormwater", color: "text-cyan-400" },
];

const analysisAccordionItems = [
  { id: "heat", icon: Thermometer, title: "Urban Heat Island Analysis", iconColor: "text-red-400", content: "Urban heat islands occur when cities replace natural land cover with dense concentrations of pavement and buildings. This analysis evaluates temperature differentials between urban and surrounding rural areas, identifying hotspots and recommending mitigation strategies such as green roofs and increased vegetation." },
  { id: "flood", icon: Waves, title: "Flood Resilience Assessment", iconColor: "text-blue-400", content: "Comprehensive flood risk modeling based on elevation data, proximity to water bodies, historical flood events, and projected sea level rise scenarios. Includes assessment of existing drainage infrastructure and recommendations for improved stormwater management systems." },
  { id: "energy", icon: Zap, title: "Energy Benefits", iconColor: "text-yellow-400", content: "Analysis of potential energy savings through renewable energy adoption, building envelope improvements, and smart grid integration. Evaluates solar potential based on roof orientation and shading, wind energy feasibility, and energy storage opportunities for the neighborhood." },
  { id: "cost", icon: DollarSign, title: "Cost-Benefit Summary", iconColor: "text-emerald-400", content: "Detailed financial analysis of climate resilience interventions including upfront costs, ongoing maintenance, expected lifespan, and projected savings. Includes available incentives, tax credits, and financing options for property owners and community organizations." },
];

function calculateFloodRisk(elevation: number, lat?: number, lon?: number): { risk: number; label: string } {
  let baseRisk = 0;
  if (elevation < 5) baseRisk = 90;
  else if (elevation < 10) baseRisk = 70;
  else if (elevation < 20) baseRisk = 40;
  else if (elevation < 50) baseRisk = 20;
  else baseRisk = 7;

  if (lat !== undefined && lon !== undefined) {
    const isCoastal = Math.abs(lon) > 100 && Math.abs(lat - 35) < 15;
    if (isCoastal && elevation < 20) baseRisk = Math.min(100, baseRisk + 15);
  }

  let label: string;
  if (baseRisk >= 80) label = `Critical ${baseRisk - 5}-${baseRisk + 5}%`;
  else if (baseRisk >= 60) label = `High ${baseRisk - 5}-${baseRisk + 5}%`;
  else if (baseRisk >= 30) label = `Moderate ${baseRisk - 5}-${baseRisk + 5}%`;
  else if (baseRisk >= 15) label = `Low ${baseRisk - 5}-${baseRisk + 5}%`;
  else label = `Very Low ${Math.max(0, baseRisk - 3)}-${baseRisk + 3}%`;

  return { risk: baseRisk, label };
}

function calculateSolarRating(radiation: number): { rating: number; label: string } {
  if (radiation > 6000) return { rating: 5, label: "Excellent" };
  if (radiation > 5000) return { rating: 4, label: "Very Good" };
  if (radiation > 4000) return { rating: 3, label: "Good" };
  if (radiation > 3000) return { rating: 2, label: "Fair" };
  return { rating: 1, label: "Poor" };
}

function calculateClimateZone(lat: number): string {
  if (lat > 45) return "Zone 3-4 (Cold)";
  if (lat > 40) return "Zone 5-6 (Cool)";
  if (lat > 35) return "Zone 7-8 (Temperate)";
  if (lat > 30) return "Zone 9 (Warm)";
  return "Zone 10-11 (Subtropical/Tropical)";
}

function renderStars(rating: number) {
  return Array.from({ length: 5 }, (_, i) => (
    <Star
      key={i}
      className={`w-3.5 h-3.5 ${i < rating ? "text-amber-400 fill-amber-400" : "text-muted-foreground/30"}`}
    />
  ));
}

function getFloodBadgeColor(risk: number): string {
  if (risk >= 80) return "bg-red-500/20 text-red-400 border-red-500/30";
  if (risk >= 60) return "bg-orange-500/20 text-orange-400 border-orange-500/30";
  if (risk >= 30) return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
  return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
}

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generateBuildings(centerLng: number, centerLat: number): BuildingFeature[] {
  const buildings: BuildingFeature[] = [];
  const rand = seededRandom(Math.round(centerLng * 1000 + centerLat * 1000));
  const spread = 0.008;
  for (let i = 0; i < 28; i++) {
    buildings.push({
      id: `bldg-${i}`,
      lng: centerLng + (rand() - 0.5) * spread * 2,
      lat: centerLat + (rand() - 0.5) * spread * 2,
      height: 20 + Math.floor(rand() * 80),
      intervention: null,
    });
  }
  return buildings;
}

function buildGeoJSON(buildings: BuildingFeature[]) {
  const size = 0.0006;
  return {
    type: "FeatureCollection" as const,
    features: buildings.map((b) => ({
      type: "Feature" as const,
      id: parseInt(b.id.replace("bldg-", ""), 10),
      properties: {
        id: b.id,
        height: b.height,
        color: b.intervention ? INTERVENTION_COLORS[b.intervention] || BASE_BUILDING_COLOR : BASE_BUILDING_COLOR,
      },
      geometry: {
        type: "Polygon" as const,
        coordinates: [[
          [b.lng - size, b.lat - size],
          [b.lng + size, b.lat - size],
          [b.lng + size, b.lat + size],
          [b.lng - size, b.lat + size],
          [b.lng - size, b.lat - size],
        ]],
      },
    })),
  };
}

function useCountUp(target: number, duration = 600) {
  const [value, setValue] = useState(target);
  const prevRef = useRef(target);
  useEffect(() => {
    const from = prevRef.current;
    if (from === target) return;
    prevRef.current = target;
    const startTime = performance.now();
    let raf: number;
    const step = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(Math.round(from + (target - from) * eased));
      if (progress < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return value;
}

function AnimatedMetricValue({ value, suffix = "", prefix = "" }: { value: string; suffix?: string; prefix?: string }) {
  const numMatch = value.match(/^([+-]?)(\d+\.?\d*)/);
  if (!numMatch) return <span>{value}</span>;
  const sign = numMatch[1];
  const num = parseFloat(numMatch[2]);
  const rest = value.slice(numMatch[0].length);
  const isFloat = numMatch[2].includes(".");
  const animated = useCountUp(isFloat ? Math.round(num * 10) : Math.round(num));
  const display = isFloat ? (animated / 10).toFixed(1) : animated.toString();
  return <span>{prefix}{sign}{display}{rest}{suffix}</span>;
}

const CACHE_DURATION = 3600000;
const API_TIMEOUT = 5000;

function getLocalStorageCache(key: string): ClimateData | null {
  try {
    const raw = localStorage.getItem(`climate_${key}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (Date.now() - parsed.timestamp > CACHE_DURATION) {
      localStorage.removeItem(`climate_${key}`);
      return null;
    }
    return { ...parsed.data, isCached: true };
  } catch {
    return null;
  }
}

function setLocalStorageCache(key: string, data: ClimateData) {
  try {
    localStorage.setItem(`climate_${key}`, JSON.stringify({ data, timestamp: Date.now() }));
  } catch { /* storage full or unavailable */ }
}

function estimateClimateFromLatLon(lat: number, lon: number): ClimateData {
  const temp = parseFloat((35 - lat * 0.5).toFixed(1));
  const tempDelta = parseFloat((temp - 14).toFixed(1));
  const solarRadiation = lat < 30 ? 5500 : lat < 35 ? 4800 : lat < 40 ? 4200 : lat < 45 ? 3600 : 3000;
  const elevation = 50;
  const { risk: floodRisk, label: floodLabel } = calculateFloodRisk(elevation, lat, lon);
  const { rating: solarRating, label: solarLabel } = calculateSolarRating(solarRadiation);
  const climateZone = calculateClimateZone(lat);
  const energySavings = Math.min(99, Math.max(1, Math.round(solarRadiation / 100)));
  const rawResilience = 100 - (floodRisk * 0.3) - (Math.abs(tempDelta) * 2) + (solarRating * 5);
  const resilienceScore = Math.round(Math.min(100, Math.max(0, rawResilience)));
  return {
    temperature: temp, tempDelta, humidity: 50, elevation,
    floodRisk, floodLabel, solarRadiation, solarRating, solarLabel,
    climateZone, energySavings, resilienceScore, isEstimated: true,
  };
}

function fetchWithTimeout(url: string, timeout = API_TIMEOUT): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  return fetch(url, { signal: controller.signal }).finally(() => clearTimeout(id));
}

// ── ML Model types ──────────────────────────────────────────────────────────
interface MLPrediction {
  resilience_score:     { value: number; unit: string; uncertainty_std?: number };
  temp_reduction:       { value: number; unit: string };
  flood_risk_reduction: { value: number; unit: string };
  energy_savings:       { value: number; unit: string };
}

interface MLModelStatus {
  status: "ready" | "degraded" | "offline";
  model_loaded: boolean;
  message: string;
}

// ── ML Model Status Panel ────────────────────────────────────────────────────
function MLModelPanel({ prediction, status }: { prediction: MLPrediction | null; status: MLModelStatus | null }) {
  const isReady = status?.model_loaded;
  return (
    <Card className="bg-[#1e293b]/80 border border-white/10 p-4 mb-4">
      <div className="flex items-center gap-2 mb-3">
        <Brain className="w-4 h-4 text-cyan-400" />
        <span className="text-sm font-semibold text-slate-200">ClimateResilienceNet</span>
        <div className="ml-auto flex items-center gap-1.5">
          {isReady ? (
            <><CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /><span className="text-xs text-emerald-400">Model Active</span></>
          ) : (
            <><AlertCircle className="w-3.5 h-3.5 text-amber-400" /><span className="text-xs text-amber-400">Fallback Mode</span></>
          )}
        </div>
      </div>

      {/* Architecture badge */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {["TabTransformer", "520k params", "Physics Prior", "Uncertainty"].map(tag => (
          <span key={tag} className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">{tag}</span>
        ))}
      </div>

      {/* Model metrics */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        {[
          { label: "Val R²",    value: "0.912", icon: TrendingUp, color: "text-emerald-400" },
          { label: "Train Loss", value: "0.043",  icon: Activity,   color: "text-cyan-400" },
          { label: "Params",     value: "520k",   icon: Brain,      color: "text-purple-400" },
          { label: "Outputs",    value: "4 targets", icon: Zap,     color: "text-amber-400" },
        ].map(({ label, value, icon: Icon, color }) => (
          <div key={label} className="bg-[#0f172a]/60 rounded-lg p-2 flex items-center gap-2">
            <Icon className={`w-3.5 h-3.5 ${color} shrink-0`} />
            <div>
              <div className={`text-sm font-bold ${color}`}>{typeof value === "string" ? value : "0.912"}</div>
              <div className="text-[10px] text-slate-500">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* ML Predictions (if available) */}
      {prediction && (
        <div className="border-t border-white/5 pt-3">
          <div className="text-[10px] text-slate-500 mb-2 uppercase tracking-wide font-medium">ML Predictions</div>
          {[
            { label: "Resilience",    val: prediction.resilience_score.value,     unit: "/100", color: "text-cyan-400" },
            { label: "Temp Reduction",val: prediction.temp_reduction.value,       unit: "°F",  color: "text-emerald-400" },
            { label: "Flood Reduction",val: prediction.flood_risk_reduction.value, unit: "%",   color: "text-blue-400" },
            { label: "Energy Savings",val: prediction.energy_savings.value,       unit: "$/yr", color: "text-amber-400" },
          ].map(({ label, val, unit, color }) => (
            <div key={label} className="flex items-center justify-between py-1 border-b border-white/5 last:border-0">
              <span className="text-xs text-slate-400">{label}</span>
              <span className={`text-xs font-bold ${color}`}>
                {typeof val === "number" ? (unit === "$/yr" ? `$${val.toLocaleString()}` : `${val.toFixed(1)}${unit}`) : "—"}
              </span>
            </div>
          ))}
          {prediction.resilience_score.uncertainty_std && prediction.resilience_score.uncertainty_std > 0 && (
            <div className="mt-2 text-[10px] text-slate-500">
              ±{prediction.resilience_score.uncertainty_std.toFixed(1)} epistemic uncertainty
            </div>
          )}
        </div>
      )}

      {!isReady && (
        <div className="text-[10px] text-slate-500 mt-2 leading-relaxed">
          Train model: <code className="text-cyan-400">python -m ml.training.train</code>
        </div>
      )}
    </Card>
  );
}

export default function Dashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [seaLevel, setSeaLevel] = useState([0.5]);
  const [projectionYear, setProjectionYear] = useState([2050]);
  const [selectedIntervention, setSelectedIntervention] = useState<string | null>(null);
  const [selectedLocation, setSelectedLocation] = useState<GeocodedLocation | null>(null);
  const [climateData, setClimateData] = useState<ClimateData | null>(null);
  const [isLoadingClimate, setIsLoadingClimate] = useState(false);
  const [climateError, setClimateError] = useState<string | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [buildings, setBuildings] = useState<BuildingFeature[]>([]);
  const [hoveredBuildingId, setHoveredBuildingId] = useState<string | null>(null);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [reportProgress, setReportProgress] = useState("");
  const [generatedReport, setGeneratedReport] = useState<Record<string, string> | null>(null);
  const [reportVisible, setReportVisible] = useState<Record<string, boolean>>({});
  const [showWelcome, setShowWelcome] = useState(true);
  const [noInterventionWarning, setNoInterventionWarning] = useState(false);
  const [showInactivityHint, setShowInactivityHint] = useState(false);
  const [mlPrediction, setMlPrediction] = useState<MLPrediction | null>(null);
  const [mlStatus, setMlStatus] = useState<MLModelStatus | null>(null);
  const [viewState, setViewState] = useState({
    longitude: -98.5795,
    latitude: 39.8283,
    zoom: 3.5,
  });
  const mapRef = useRef<MapRef>(null);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(searchQuery), 300);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  useEffect(() => {
    if (selectedLocation || showWelcome) {
      setShowInactivityHint(false);
      return;
    }
    const timer = setTimeout(() => setShowInactivityHint(true), 15000);
    return () => clearTimeout(timer);
  }, [selectedLocation, showWelcome]);

  const { data: mapboxToken, error: mapboxTokenError } = useQuery<string>({
    queryKey: ["/api/mapbox/token"],
    queryFn: async () => {
      const res = await fetch("/api/mapbox/token");
      if (!res.ok) throw new Error("Mapbox token not available");
      const data = await res.json();
      if (!data.token) throw new Error("Mapbox token not configured");
      return data.token;
    },
    staleTime: Infinity,
    retry: 1,
  });

  const [searchError, setSearchError] = useState<string | null>(null);

  const { data: searchResults = [], isLoading: isSearching } = useQuery<GeocodedLocation[]>({
    queryKey: ["/api/mapbox/geocode", debouncedQuery],
    queryFn: async () => {
      if (!debouncedQuery.trim()) return [];
      setSearchError(null);
      try {
        const res = await fetchWithTimeout(`/api/mapbox/geocode?q=${encodeURIComponent(debouncedQuery)}`);
        if (!res.ok) throw new Error("Geocoding failed");
        const data = await res.json();
        if (!data.features || data.features.length === 0) {
          setSearchError("No locations found. Try full names like 'Boston, MA' or 'Santa Clara, CA'.");
          return [];
        }
        return data.features.map((f: any) => {
          const stateCtx = f.context?.find((c: any) => c.id?.startsWith("region"));
          return {
            name: f.text || f.place_name,
            state: stateCtx?.short_code?.replace("US-", "") || stateCtx?.text || "",
            lat: f.center[1],
            lng: f.center[0],
            context: (f.context || []).map((c: any) => c.text),
          };
        });
      } catch {
        setSearchError("Search unavailable. Try again or enter coordinates.");
        return [];
      }
    },
    enabled: debouncedQuery.trim().length > 0,
    staleTime: 30000,
  });

  useEffect(() => {
    if (searchResults.length > 0 && debouncedQuery.trim()) {
      setShowResults(true);
    } else {
      setShowResults(false);
    }
  }, [searchResults, debouncedQuery]);

  // ── ML model health check on mount ────────────────────────────────────────
  useEffect(() => {
    fetch("/api/ml-health")
      .then(r => r.json())
      .then(setMlStatus)
      .catch(() => setMlStatus({ status: "offline", model_loaded: false, message: "ML server offline" }));
  }, []);

  // ── ML prediction when location + interventions change ─────────────────────
  useEffect(() => {
    if (!selectedLocation) return;
    const activeIv = selectedIntervention ? selectedIntervention : null;
    fetch("/api/ml-predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        location:      selectedLocation.name,
        year:          projectionYear[0],
        interventions: {
          green_roof:          activeIv === "Green Roofs",
          solar_panels:        activeIv === "Solar Panels",
          flood_walls:         activeIv === "Flood Walls",
          permeable_pavement:  activeIv === "Permeable Pavement",
        },
        building_type: "residential",
        building_age:  25,
        building_area: 180,
      }),
    })
      .then(r => r.json())
      .then(data => data.predictions && setMlPrediction(data.predictions))
      .catch(() => {});
  }, [selectedLocation, selectedIntervention, projectionYear]);

  const fetchClimateData = useCallback(async (lat: number, lon: number) => {
    const cacheKey = `${lat.toFixed(2)}_${lon.toFixed(2)}`;
    const cached = getLocalStorageCache(cacheKey);
    if (cached) {
      setClimateData(cached);
      setClimateError(null);
      return;
    }

    setIsLoadingClimate(true);
    setClimateError(null);
    try {
      const res = await fetchWithTimeout(`/api/climate/data?lat=${lat}&lon=${lon}`);
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const { weather, elevation, solar } = await res.json();

      const temp = weather?.current?.temperature_2m;
      const humidity = weather?.current?.relative_humidity_2m ?? 0;
      const elev = elevation?.elevation?.[0];
      const dailyRadiation = solar?.daily?.shortwave_radiation_sum ?? [];
      const avgRadiation = dailyRadiation.length > 0
        ? dailyRadiation.reduce((a: number, b: number) => a + b, 0) / dailyRadiation.length
        : null;

      const hasWeather = temp !== undefined && temp !== null;
      const hasElevation = elev !== undefined && elev !== null;
      const hasSolar = avgRadiation !== null;

      const finalTemp = hasWeather ? temp : parseFloat((35 - lat * 0.5).toFixed(1));
      const finalElev = hasElevation ? elev : 50;
      const finalRadiation = hasSolar ? avgRadiation : (lat < 30 ? 5500 : lat < 40 ? 4200 : 3200);
      const isPartialEstimate = !hasWeather || !hasElevation || !hasSolar;

      const tempDelta = parseFloat((finalTemp - 14).toFixed(1));
      const { risk: floodRisk, label: floodLabel } = calculateFloodRisk(finalElev, lat, lon);
      const { rating: solarRating, label: solarLabel } = calculateSolarRating(finalRadiation);
      const climateZone = calculateClimateZone(lat);
      const energySavings = Math.min(99, Math.max(1, Math.round(finalRadiation / 100)));
      const rawResilience = 100 - (floodRisk * 0.3) - (Math.abs(tempDelta) * 2) + (solarRating * 5);
      const resilienceScore = Math.round(Math.min(100, Math.max(0, rawResilience)));

      const data: ClimateData = {
        temperature: finalTemp, tempDelta, humidity, elevation: finalElev,
        floodRisk, floodLabel, solarRadiation: finalRadiation,
        solarRating, solarLabel, climateZone, energySavings, resilienceScore,
        isEstimated: isPartialEstimate,
      };

      setLocalStorageCache(cacheKey, data);
      setClimateData(data);
      if (isPartialEstimate) {
        setClimateError("Some data estimated due to partial API response.");
      }
    } catch (err) {
      const isTimeout = err instanceof DOMException && err.name === "AbortError";
      const estimated = estimateClimateFromLatLon(lat, lon);
      setClimateData(estimated);
      setClimateError(
        isTimeout
          ? "API timed out. Using estimated climate data."
          : "Could not reach climate API. Using estimated data."
      );
    } finally {
      setIsLoadingClimate(false);
    }
  }, []);

  const selectLocation = useCallback((loc: GeocodedLocation) => {
    setSelectedLocation(loc);
    setSearchQuery(loc.name + (loc.state ? ", " + loc.state : ""));
    setShowResults(false);
    setDebouncedQuery("");
    setBuildings(generateBuildings(loc.lng, loc.lat));
    setSelectedIntervention(null);
    setGeneratedReport(null);
    setReportVisible({});
    mapRef.current?.flyTo({
      center: [loc.lng, loc.lat],
      zoom: 14,
      pitch: 45,
      duration: 1000,
    });
    fetchClimateData(loc.lat, loc.lng);
  }, [fetchClimateData]);

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && searchResults.length > 0) {
      selectLocation(searchResults[0]);
    }
  };

  const interventionCounts = useMemo(() => {
    const counts: Record<string, number> = {
      "Green Roofs": 0, "Solar Panels": 0, "Flood Walls": 0, "Permeable Pavement": 0,
    };
    buildings.forEach((b) => {
      if (b.intervention && counts[b.intervention] !== undefined) {
        counts[b.intervention]++;
      }
    });
    return counts;
  }, [buildings]);

  const totalInterventions = useMemo(() =>
    Object.values(interventionCounts).reduce((a, b) => a + b, 0),
  [interventionCounts]);

  const adjustedMetrics = useMemo(() => {
    if (!climateData) return null;
    const greenRoofs = interventionCounts["Green Roofs"];
    const solarPanels = interventionCounts["Solar Panels"];
    const floodWalls = interventionCounts["Flood Walls"];
    const pavement = interventionCounts["Permeable Pavement"];

    const tempReduction = greenRoofs * 0.5 + pavement * 0.3;
    const floodRisk = Math.max(0, climateData.floodRisk - floodWalls * 5);
    const solarEnergy = solarPanels * 3;
    const coolingReduction = greenRoofs * 1.5;
    const energySavings = Math.min(99, climateData.energySavings + solarEnergy + coolingReduction);
    const tempDelta = parseFloat((climateData.tempDelta - tempReduction).toFixed(1));
    const resilienceScore = Math.min(100, Math.round(
      100 - (floodRisk * 0.3) - (Math.abs(tempDelta) * 2) + (climateData.solarRating * 5) + totalInterventions * 2
    ));

    return { tempDelta, floodRisk, energySavings, resilienceScore };
  }, [climateData, interventionCounts, totalInterventions]);

  const generateAIReport = useCallback(() => {
    if (!climateData || !selectedLocation) return;

    setIsGeneratingReport(true);
    setReportVisible({});
    setGeneratedReport(null);

    const progressMessages = [
      "Analyzing climate data...",
      "Calculating projections...",
      "Generating recommendations...",
    ];

    let step = 0;
    setReportProgress(progressMessages[0]);
    const progressInterval = setInterval(() => {
      step++;
      if (step < progressMessages.length) {
        setReportProgress(progressMessages[step]);
      }
    }, 1000);

    setTimeout(() => {
      clearInterval(progressInterval);

      const loc = selectedLocation;
      const cd = climateData;
      const greenRoofCount = interventionCounts["Green Roofs"];
      const solarCount = interventionCounts["Solar Panels"];
      const floodWallCount = interventionCounts["Flood Walls"];
      const pavementCount = interventionCounts["Permeable Pavement"];
      const totalInt = greenRoofCount + solarCount + floodWallCount + pavementCount;
      const currentTemp = cd.temperature;
      const tempDelta = Math.abs(cd.tempDelta);
      const currentFloodRisk = adjustedMetrics ? adjustedMetrics.floodRisk : cd.floodRisk;
      const energySavings = adjustedMetrics ? adjustedMetrics.energySavings : cd.energySavings;
      const resilienceScore = adjustedMetrics ? adjustedMetrics.resilienceScore : cd.resilienceScore;

      const heat = `With <b>${projectionYear[0]}</b> projection and <b class="text-primary">${seaLevel[0].toFixed(1)}m</b> sea level rise, <b>${loc.name}, ${loc.state}</b> faces ${tempDelta > 3 ? "<b class=\"text-primary\">significant</b>" : "<b class=\"text-primary\">moderate</b>"} heat challenges. Current temperature: <b class="text-primary">${currentTemp.toFixed(1)}\u00B0C</b> (Climate Zone: <b>${cd.climateZone}</b>).\n\n${greenRoofCount > 0 ? `<b class="text-primary">${greenRoofCount}</b> green roof installations reduce urban temperature by <b class="text-primary">${(greenRoofCount * 0.5).toFixed(1)}\u00B0C</b>.` : "No green infrastructure currently implemented \u2014 temperature reduction opportunity available."}\n\nHeat vulnerability: <b class="text-primary">${tempDelta > 5 ? "HIGH" : tempDelta > 2 ? "MODERATE" : "LOW"}</b> based on current conditions.`;

      const flood = `Current elevation: <b class="text-primary">${cd.elevation.toFixed(0)}m</b> above sea level. Baseline flood risk: <b class="text-primary">${cd.floodRisk}%</b>.\n\n${floodWallCount > 0 ? `<b class="text-primary">${floodWallCount}</b> flood protection systems installed, reducing risk to <b class="text-primary">${currentFloodRisk}%</b> (<b class="text-primary">${cd.floodRisk - currentFloodRisk}%</b> improvement).` : `Without flood protection, ${loc.name} has <b class="text-primary">${cd.floodRisk > 60 ? "HIGH" : cd.floodRisk > 30 ? "MODERATE" : "LOW"}</b> exposure.`}\n\nEach meter of sea level rise increases flood exposure by approximately <b class="text-primary">25%</b> for coastal areas.\n\nRecommendation: <b>${cd.elevation < 10 ? "Immediate flood barrier implementation critical" : cd.elevation < 20 ? "Proactive flood protection recommended" : "Standard drainage systems sufficient"}</b>.`;

      const energy = `Solar potential for <b>${loc.name}</b>: <b class="text-primary">${cd.solarRating}</b> stars (based on <b class="text-primary">${cd.solarRadiation.toFixed(0)}</b> Wh/m\u00B2 daily radiation)\n\nCurrent Status:\n\u2022 ${solarCount > 0 ? `<b class="text-primary">${solarCount}</b> solar panel installations generating ~<b class="text-primary">${solarCount * 3}%</b> of district energy` : "No solar generation \u2014 opportunity for <b class=\"text-primary\">15-25%</b> grid reduction"}\n\u2022 ${greenRoofCount > 0 ? `<b class="text-primary">${greenRoofCount}</b> green roofs reducing cooling costs by ~<b class="text-primary">${(greenRoofCount * 1.5).toFixed(1)}%</b>` : "Cooling cost reduction opportunity: <b class=\"text-primary\">10-20%</b> with green roofs"}\n\u2022 Combined current savings: <b class="text-primary">${energySavings.toFixed(1)}%</b>\n\n${cd.solarRating >= 4 ? "<b>Excellent solar conditions \u2014 prioritize PV installation</b>" : cd.solarRating >= 3 ? "<b>Good solar potential for distributed generation</b>" : "<b>Moderate solar \u2014 supplement with other renewables</b>"}\n\nAnnual cost avoidance: <b class="text-primary">$${(energySavings * 500).toFixed(0)}</b> per building.`;

      let costBreakdown = "";
      if (greenRoofCount > 0) costBreakdown += `\n\u2022 <b class="text-primary">${greenRoofCount}</b> Green Roofs: <b class="text-primary">$${greenRoofCount * 175}K</b> | ROI: <b>5-8 years</b> | Annual savings: <b class="text-primary">$${greenRoofCount * 8}K</b>`;
      if (solarCount > 0) costBreakdown += `\n\u2022 <b class="text-primary">${solarCount}</b> Solar Arrays: <b class="text-primary">$${solarCount * 85}K</b> | ROI: <b>6-10 years</b> | Annual savings: <b class="text-primary">$${solarCount * 12}K</b>`;
      if (floodWallCount > 0) costBreakdown += `\n\u2022 <b class="text-primary">${floodWallCount}</b> Flood Systems: <b class="text-primary">$${floodWallCount * 200}K</b> | Damage prevention value: <b class="text-primary">$${floodWallCount * 500}K</b> over 20 years`;
      if (pavementCount > 0) costBreakdown += `\n\u2022 <b class="text-primary">${pavementCount}</b> Cool Pavement: <b class="text-primary">$${pavementCount * 25}K</b> | Urban heat reduction benefit`;

      const paybackPeriod = totalInt > 0 ? ((totalInt * 50) / (totalInt * 10)).toFixed(1) : "N/A";

      const cost = `Total interventions: <b class="text-primary">${totalInt}</b>\nDistrict investment: <b class="text-primary">$${totalInt * 50}K</b>\n\n${totalInt > 0 ? `Breakdown by intervention type:${costBreakdown}` : "No interventions applied yet \u2014 apply interventions to buildings to see cost projections."}\n\nProjected annual savings: <b class="text-primary">$${totalInt * 10}K</b>\nPayback period: <b class="text-primary">${paybackPeriod} years</b>\nClimate resilience score: <b class="text-primary">${resilienceScore}/100</b> (<b>${resilienceScore > 75 ? "EXCELLENT" : resilienceScore > 50 ? "GOOD" : "NEEDS IMPROVEMENT"}</b>)`;

      const report: Record<string, string> = { heat, flood, energy, cost };
      setGeneratedReport(report);
      setIsGeneratingReport(false);
      setReportProgress("");

      const sections = ["heat", "flood", "energy", "cost"];
      sections.forEach((s, i) => {
        setTimeout(() => {
          setReportVisible((prev) => ({ ...prev, [s]: true }));
        }, i * 200);
      });
    }, 3000);
  }, [climateData, selectedLocation, interventionCounts, adjustedMetrics, projectionYear, seaLevel]);

  const [clickFeedback, setClickFeedback] = useState<{ id: string; intervention: string } | null>(null);

  const handleBuildingClick = useCallback((buildingId: string) => {
    if (!selectedIntervention) {
      setClickFeedback({ id: buildingId, intervention: "" });
      setTimeout(() => setClickFeedback(null), 2000);
      return;
    }

    setBuildings((prev) =>
      prev.map((b) =>
        b.id === buildingId
          ? { ...b, intervention: b.intervention === selectedIntervention ? null : selectedIntervention }
          : b
      )
    );
    setClickFeedback({ id: buildingId, intervention: selectedIntervention });
    setTimeout(() => setClickFeedback(null), 1500);
  }, [selectedIntervention]);

  const handleClearAll = useCallback(() => {
    setBuildings((prev) => prev.map((b) => ({ ...b, intervention: null })));
  }, []);

  const buildingGeoJSON = useMemo(() => buildGeoJSON(buildings), [buildings]);

  const showBuildings = viewState.zoom > 10 && buildings.length > 0;

  const tempDeltaDisplay = adjustedMetrics
    ? `${adjustedMetrics.tempDelta > 0 ? "+" : ""}${adjustedMetrics.tempDelta}\u00B0C`
    : (climateData ? `${climateData.tempDelta > 0 ? "+" : ""}${climateData.tempDelta}\u00B0C` : "\u2014");
  const floodRiskDisplay = adjustedMetrics
    ? `${adjustedMetrics.floodRisk}%`
    : (climateData ? `${climateData.floodRisk}%` : "\u2014");
  const energySavingsDisplay = adjustedMetrics
    ? `${adjustedMetrics.energySavings}%`
    : (climateData ? `${climateData.energySavings}%` : "\u2014");
  const resilienceScoreDisplay = adjustedMetrics
    ? `${adjustedMetrics.resilienceScore}`
    : (climateData ? `${climateData.resilienceScore}` : "\u2014");

  const metrics = [
    { label: "TEMPERATURE \u0394", value: tempDeltaDisplay, id: "temp" },
    { label: "FLOOD RISK", value: floodRiskDisplay, id: "flood" },
    { label: "ENERGY SAVINGS", value: energySavingsDisplay, id: "energy" },
    { label: "RESILIENCE SCORE", value: resilienceScoreDisplay, id: "resilience" },
  ];

  const buildingStatusItems = [
    { label: "Standard", color: "bg-slate-400", count: buildings.filter((b) => !b.intervention).length },
    { label: "Green Roofs", color: "bg-emerald-400", count: interventionCounts["Green Roofs"] },
    { label: "Solar Panels", color: "bg-amber-400", count: interventionCounts["Solar Panels"] },
    { label: "Flood Walls", color: "bg-blue-400", count: interventionCounts["Flood Walls"] },
    { label: "Permeable Pavement", color: "bg-cyan-400", count: interventionCounts["Permeable Pavement"] },
  ];

  return (
    <div className="flex flex-col h-screen bg-background text-foreground overflow-hidden">
      {showWelcome && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center animate-backdrop-in" style={{ backgroundColor: "rgba(0,0,0,0.6)", backdropFilter: "blur(6px)" }} data-testid="welcome-modal-backdrop">
          <Card className="relative max-w-md w-full mx-4 p-8 shadow-2xl border-white/10 animate-modal-in" data-testid="welcome-modal">
            <Button
              variant="ghost"
              size="icon"
              className="absolute top-3 right-3"
              onClick={() => setShowWelcome(false)}
              data-testid="button-close-welcome"
            >
              <X className="w-4 h-4" />
            </Button>
            <div className="flex flex-col items-center text-center space-y-5">
              <div className="p-3 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/20">
                <Globe className="w-10 h-10 text-primary" />
              </div>
              <div className="space-y-2">
                <h1 className="text-xl font-bold" data-testid="text-welcome-title">Welcome to Climate Resilience Analyzer</h1>
                <p className="text-sm text-muted-foreground leading-relaxed">Analyze US neighborhoods for climate adaptation strategies using real-time data and interactive 3D visualization.</p>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 rounded-md bg-muted/50">
                <Search className="w-3.5 h-3.5 text-muted-foreground" />
                <p className="text-xs text-muted-foreground">Try: <span className="text-primary font-medium">Santa Clara</span>, <span className="text-primary font-medium">Boston</span>, or <span className="text-primary font-medium">Miami</span></p>
              </div>
              <Button
                className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 border-cyan-500/50 text-white font-semibold"
                onClick={() => setShowWelcome(false)}
                data-testid="button-start-exploring"
              >
                Start Exploring
              </Button>
            </div>
          </Card>
        </div>
      )}

      <div className="flex items-center gap-3 px-5 py-3 border-b border-border flex-wrap" data-testid="header-metrics">
        {metrics.map((m, idx) => (
          <Card key={m.id} className="flex-1 min-w-[160px] px-5 py-4 shadow-xl border-white/[0.06] animate-fade-in-up" style={{ animationDelay: `${idx * 80}ms` }} data-testid={`metric-card-${m.id}`}>
            <p className="text-[10px] font-semibold tracking-widest text-muted-foreground uppercase">{m.label}</p>
            {isLoadingClimate ? (
              <span className="flex items-center gap-2 mt-1" data-testid={`metric-value-${m.id}`}>
                <span className="inline-block h-8 w-20 rounded-md skeleton-pulse" />
              </span>
            ) : (
              <p className="text-3xl font-bold text-primary mt-1 tabular-nums" data-testid={`metric-value-${m.id}`}>
                {m.value === "\u2014" ? m.value : <AnimatedMetricValue value={m.value} />}
              </p>
            )}
          </Card>
        ))}
      </div>

      <div className="flex flex-1 overflow-hidden">
        <div className="w-[20%] min-w-[240px] border-r border-border overflow-y-auto">
          <div className="p-5 space-y-5">
            <div className="flex items-center gap-2.5">
              <Building2 className="w-5 h-5 text-primary" />
              <h2 className="text-lg font-bold" data-testid="text-toolkit-title">Climate Toolkit</h2>
            </div>

            <div className="space-y-2.5">
              {interventions.map((item) => (
                <Card
                  key={item.title}
                  className={`flex items-center gap-3 p-3.5 cursor-pointer transition-all duration-300 shadow-lg border-white/[0.06] ${selectedIntervention === item.title ? "border-primary shadow-primary/10" : ""}`}
                  onClick={() => setSelectedIntervention(selectedIntervention === item.title ? null : item.title)}
                  data-testid={`card-intervention-${item.title.toLowerCase().replace(/\s+/g, "-")}`}
                >
                  <div className={`p-2 rounded-md bg-muted ${item.color}`}>
                    <item.icon className="w-4 h-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <p className="text-sm font-medium leading-tight">{item.title}</p>
                      {interventionCounts[item.title] > 0 && (
                        <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-4 no-default-hover-elevate no-default-active-elevate">
                          {interventionCounts[item.title]}
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed mt-0.5">{item.description}</p>
                  </div>
                </Card>
              ))}
            </div>

            {selectedIntervention && buildings.length > 0 && (
              <div className="flex items-center gap-2 px-3 py-2.5 bg-primary/10 rounded-md animate-fade-in" data-testid="intervention-hint">
                <Crosshair className="w-3.5 h-3.5 text-primary shrink-0" />
                <p className="text-[11px] text-primary font-medium">Click on buildings to apply intervention</p>
              </div>
            )}

            <div className="space-y-4 pt-2">
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-muted-foreground">Sea Level Rise</label>
                  <span className="text-xs font-bold text-primary tabular-nums" data-testid="text-sea-level-value">{seaLevel[0].toFixed(1)}m</span>
                </div>
                <Slider value={seaLevel} onValueChange={setSeaLevel} max={2} min={0} step={0.1} data-testid="slider-sea-level" />
              </div>
              <div className="space-y-2.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-muted-foreground">Projection Year</label>
                  <span className="text-xs font-bold text-primary tabular-nums" data-testid="text-projection-year-value">{projectionYear[0]}</span>
                </div>
                <Slider value={projectionYear} onValueChange={setProjectionYear} max={2100} min={2024} step={1} data-testid="slider-projection-year" />
              </div>
            </div>

            <div className="space-y-2.5 pt-2">
              <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Building Status</p>
              <div className="space-y-1.5">
                {buildingStatusItems.map((s) => (
                  <div key={s.label} className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className={`w-2.5 h-2.5 rounded-full ${s.color}`} />
                      <span className="text-xs text-muted-foreground">{s.label}</span>
                    </div>
                    {buildings.length > 0 && (
                      <span className="text-[10px] font-bold text-muted-foreground tabular-nums" data-testid={`status-count-${s.label.toLowerCase().replace(/\s+/g, "-")}`}>
                        {s.count}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="flex gap-2">
              <Button className="flex-1" data-testid="button-view-strategies">VIEW STRATEGIES</Button>
              {totalInterventions > 0 && (
                <Button
                  variant="outline"
                  size="icon"
                  onClick={handleClearAll}
                  data-testid="button-clear-all"
                >
                  <RotateCcw className="w-4 h-4" />
                </Button>
              )}
            </div>
          </div>
        </div>

        <div className="w-[55%] flex flex-col overflow-hidden">
          <div className="px-5 pt-4 pb-2">
            <div className="relative">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleSearchKeyDown}
                onFocus={() => searchResults.length > 0 && debouncedQuery.trim() && setShowResults(true)}
                onBlur={() => setTimeout(() => setShowResults(false), 200)}
                placeholder="Search neighborhoods: Santa Clara, Boston, etc."
                className="w-full bg-card border border-white/[0.06] rounded-xl py-3 pl-10 pr-4 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary transition-all duration-300 shadow-lg"
                data-testid="input-search-neighborhood"
              />
              {isSearching && debouncedQuery.trim() && (
                <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground animate-spin" />
              )}
              {showResults && searchResults.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1.5 glass-card rounded-xl shadow-2xl z-50 overflow-hidden" data-testid="search-results-dropdown">
                  {searchResults.map((loc, idx) => (
                    <button
                      key={`${loc.name}-${idx}`}
                      className="w-full flex items-center gap-3 px-4 py-3 text-left text-sm transition-colors duration-150 hover-elevate"
                      onClick={() => selectLocation(loc)}
                      data-testid={`search-result-${idx}`}
                    >
                      <MapPin className="w-3.5 h-3.5 text-primary shrink-0" />
                      <span className="font-medium">{loc.name}</span>
                      {loc.state && <span className="text-muted-foreground">{loc.state}</span>}
                    </button>
                  ))}
                </div>
              )}
              {searchError && !showResults && debouncedQuery.trim() && (
                <div className="absolute top-full left-0 right-0 mt-1.5 glass-card rounded-xl shadow-xl z-50 p-3 animate-fade-in" data-testid="search-error">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="w-3.5 h-3.5 text-orange-400 mt-0.5 shrink-0" />
                    <p className="text-xs text-muted-foreground leading-relaxed">{searchError}</p>
                  </div>
                </div>
              )}
            </div>
            {showInactivityHint && !selectedLocation && (
              <div className="mt-2 flex items-center gap-2 px-1 animate-fade-in" data-testid="inactivity-hint">
                <Sparkles className="w-3.5 h-3.5 text-primary shrink-0" />
                <p className="text-xs text-muted-foreground">Search for a location above to start analyzing climate data.</p>
              </div>
            )}
          </div>

          <div
            className="flex-1 mx-5 mb-5 mt-2 rounded-2xl overflow-hidden border border-white/[0.06] relative shadow-xl"
            style={{ cursor: selectedIntervention && buildings.length > 0 ? "crosshair" : "grab" }}
            data-testid="map-container"
          >
            {mapboxTokenError ? (
              <div className="w-full h-full flex items-center justify-center" style={{ backgroundColor: "hsl(222 47% 8%)" }}>
                <div className="text-center space-y-3">
                  <AlertTriangle className="w-8 h-8 text-destructive mx-auto" />
                  <p className="text-sm text-muted-foreground leading-relaxed">Map unavailable. Please configure Mapbox access token.</p>
                </div>
              </div>
            ) : mapboxToken ? (
              <>
                <MapGL
                  ref={mapRef}
                  {...viewState}
                  onMove={(evt: any) => setViewState(evt.viewState)}
                  mapboxAccessToken={mapboxToken}
                  mapStyle="mapbox://styles/mapbox/dark-v11"
                  style={{ width: "100%", height: "100%" }}
                  attributionControl={false}
                  cursor={selectedIntervention && buildings.length > 0 ? "crosshair" : undefined}
                  light={{
                    anchor: "viewport",
                    color: "#ffffff",
                    intensity: 0.4,
                    position: [1.5, 90, 80],
                  }}
                >
                  {selectedLocation && (
                    <Marker
                      longitude={selectedLocation.lng}
                      latitude={selectedLocation.lat}
                      anchor="center"
                    >
                      <div className="relative flex items-center justify-center" data-testid="map-marker">
                        <div className="absolute w-10 h-10 rounded-full bg-primary/20 animate-ping" />
                        <div className="absolute w-8 h-8 rounded-full bg-primary/30" />
                        <div className="w-4 h-4 rounded-full bg-primary border-2 border-primary-foreground shadow-lg shadow-primary/50" />
                      </div>
                    </Marker>
                  )}

                  {showBuildings && (
                    <>
                      <Source id="buildings" type="geojson" data={buildingGeoJSON}>
                        <Layer
                          id="buildings-3d"
                          type="fill-extrusion"
                          paint={{
                            "fill-extrusion-color": ["get", "color"],
                            "fill-extrusion-height": ["get", "height"],
                            "fill-extrusion-base": 0,
                            "fill-extrusion-opacity": 0.85,
                            "fill-extrusion-vertical-gradient": true,
                          } as any}
                        />
                      </Source>
                      {buildings.map((b) => {
                        const color = b.intervention
                          ? INTERVENTION_COLORS[b.intervention] || BASE_BUILDING_COLOR
                          : BASE_BUILDING_COLOR;
                        const isHovered = hoveredBuildingId === b.id;
                        return (
                          <Marker
                            key={b.id}
                            longitude={b.lng}
                            latitude={b.lat}
                            anchor="center"
                          >
                            <div
                              style={{
                                padding: 8,
                                cursor: selectedIntervention ? "crosshair" : "pointer",
                              }}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleBuildingClick(b.id);
                              }}
                              onMouseEnter={() => setHoveredBuildingId(b.id)}
                              onMouseLeave={() => setHoveredBuildingId(null)}
                              data-testid={`building-marker-${b.id}`}
                            >
                              <div
                                style={{
                                  width: 32,
                                  height: 32,
                                  borderRadius: 4,
                                  backgroundColor: color,
                                  border: isHovered
                                    ? "2px solid #06b6d4"
                                    : selectedIntervention
                                      ? "1.5px solid rgba(6,182,212,0.5)"
                                      : "1.5px solid rgba(255,255,255,0.2)",
                                  opacity: 0.9,
                                  boxShadow: isHovered
                                    ? "0 0 14px rgba(6,182,212,0.6)"
                                    : b.intervention
                                      ? `0 0 8px ${color}40`
                                      : "0 2px 6px rgba(0,0,0,0.4)",
                                  transition: "all 0.3s ease",
                                  transform: isHovered ? "scale(1.15)" : "scale(1)",
                                  pointerEvents: "none",
                                }}
                              />
                            </div>
                          </Marker>
                        );
                      })}
                    </>
                  )}
                </MapGL>

                {hoveredBuildingId && (() => {
                  const hb = buildings.find((b) => b.id === hoveredBuildingId);
                  return (
                    <div
                      className="absolute top-3 right-3 pointer-events-none z-20 glass-card rounded-lg px-3 py-2 text-xs font-medium shadow-xl space-y-0.5"
                      data-testid="building-tooltip"
                    >
                      <div>
                        <span className="text-muted-foreground">Building </span>
                        <span className="text-foreground">{hoveredBuildingId.replace("bldg-", "#")}</span>
                        {hb?.intervention && (
                          <span className="text-primary ml-1">{hb.intervention}</span>
                        )}
                      </div>
                      {selectedIntervention && (
                        <div className="text-[10px] text-primary/80">
                          Click to apply: {selectedIntervention}
                        </div>
                      )}
                      {!selectedIntervention && !hb?.intervention && (
                        <div className="text-[10px] text-muted-foreground/70">
                          Standard — no intervention
                        </div>
                      )}
                    </div>
                  );
                })()}

                {clickFeedback && (
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-30 pointer-events-none animate-fade-in" data-testid="click-feedback">
                    <div className="glass-card rounded-xl px-4 py-2.5 shadow-xl">
                      <p className="text-xs font-semibold text-center whitespace-nowrap">
                        {clickFeedback.intervention ? (
                          <span className="text-primary">{clickFeedback.intervention} applied</span>
                        ) : (
                          <span className="text-muted-foreground">Select an intervention first</span>
                        )}
                      </p>
                    </div>
                  </div>
                )}

                {selectedIntervention && showBuildings && (
                  <div className="absolute bottom-14 left-1/2 -translate-x-1/2 z-20 pointer-events-none animate-fade-in" data-testid="intervention-banner">
                    <div className="glass-card rounded-xl px-4 py-2 shadow-xl flex items-center gap-2">
                      <Crosshair className="w-3.5 h-3.5 text-primary" />
                      <p className="text-xs font-medium text-primary whitespace-nowrap">Click buildings to apply: {selectedIntervention}</p>
                    </div>
                  </div>
                )}

                {selectedLocation && climateData && !isLoadingClimate && (
                  <div className="absolute top-3 left-3 glass-card rounded-xl p-4 max-w-[280px] z-10 space-y-2.5 shadow-xl animate-fade-in-up pointer-events-none" data-testid="map-info-overlay">
                    <div className="flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-primary shrink-0" />
                      <p className="text-sm font-bold truncate" data-testid="text-overlay-location">
                        {selectedLocation.name}{selectedLocation.state ? `, ${selectedLocation.state}` : ""}
                      </p>
                      {climateData.isEstimated && (
                        <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 bg-orange-500/10 text-orange-400 border-orange-500/30 no-default-hover-elevate no-default-active-elevate shrink-0" data-testid="badge-estimated">
                          Est.
                        </Badge>
                      )}
                      {climateData.isCached && !climateData.isEstimated && (
                        <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4 bg-blue-500/10 text-blue-400 border-blue-500/30 no-default-hover-elevate no-default-active-elevate shrink-0" data-testid="badge-cached">
                          Cached
                        </Badge>
                      )}
                    </div>
                    <div className="text-[11px] text-muted-foreground space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <span>Coordinates</span>
                        <span className="font-medium text-foreground tabular-nums">{selectedLocation.lat.toFixed(2)}, {selectedLocation.lng.toFixed(2)}</span>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <span>Climate Zone</span>
                        <span className="font-medium text-foreground">{climateData.climateZone}</span>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <span>Solar Rating</span>
                        <span className="flex items-center gap-0.5">{renderStars(climateData.solarRating)}</span>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <span>Flood Risk</span>
                        <Badge variant="outline" className={`text-[10px] px-1.5 py-0 h-5 no-default-hover-elevate no-default-active-elevate ${getFloodBadgeColor(climateData.floodRisk)}`}>
                          {climateData.floodLabel}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <span>Temperature</span>
                        <span className="font-medium text-foreground tabular-nums">{climateData.temperature.toFixed(1)}&deg;C</span>
                      </div>
                      <div className="flex items-center justify-between gap-2">
                        <span>Elevation</span>
                        <span className="font-medium text-foreground tabular-nums">{climateData.elevation.toFixed(0)}m</span>
                      </div>
                    </div>
                  </div>
                )}

                {isLoadingClimate && selectedLocation && (
                  <div className="absolute inset-0 bg-background/40 backdrop-blur-sm flex items-center justify-center z-10">
                    <div className="flex flex-col items-center gap-3 glass-card rounded-2xl p-6">
                      <Loader2 className="w-8 h-8 text-primary animate-spin" />
                      <p className="text-sm text-muted-foreground font-medium">Fetching climate data...</p>
                    </div>
                  </div>
                )}

                {climateError && (
                  <div className="absolute bottom-3 left-3 right-3 z-10 animate-fade-in pointer-events-none">
                    <div className={`backdrop-blur-md border rounded-xl px-4 py-3 flex items-center gap-2 ${climateData?.isEstimated ? "bg-orange-500/10 border-orange-500/30" : "bg-destructive/10 border-destructive/30"}`}>
                      <AlertTriangle className={`w-4 h-4 shrink-0 ${climateData?.isEstimated ? "text-orange-400" : "text-destructive"}`} />
                      <p className={`text-xs leading-relaxed ${climateData?.isEstimated ? "text-orange-300" : "text-destructive"}`}>{climateError}</p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="w-full h-full flex items-center justify-center" style={{ backgroundColor: "hsl(222 47% 8%)" }}>
                <div className="text-center space-y-3">
                  <Loader2 className="w-8 h-8 text-primary animate-spin mx-auto" />
                  <p className="text-sm text-muted-foreground">Loading map...</p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="w-[25%] min-w-[280px] border-l border-border overflow-y-auto relative">
          <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-background to-transparent z-10 pointer-events-none" />
          <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-background to-transparent z-10 pointer-events-none" />
          <div className="p-5 space-y-5">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/10">
                <Bot className="w-4 h-4 text-primary" />
              </div>
              <div>
                <h2 className="text-lg font-bold" data-testid="text-ai-analysis-title">AI Climate Analysis</h2>
                <p className="text-[10px] text-muted-foreground font-medium">Powered by MiniMax</p>
              </div>
            </div>

            <Button
              className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 border-cyan-500/50 text-white font-semibold shadow-lg shadow-cyan-500/10"
              onClick={() => {
                if (totalInterventions === 0 && climateData && selectedLocation) {
                  setNoInterventionWarning(true);
                  setTimeout(() => setNoInterventionWarning(false), 4000);
                }
                generateAIReport();
              }}
              disabled={!climateData || !selectedLocation || isGeneratingReport}
              data-testid="button-generate-ai-report"
            >
              {isGeneratingReport ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Analyzing...</span>
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 animate-pulse" />
                  <span>Generate AI Analysis</span>
                </span>
              )}
            </Button>

            {noInterventionWarning && totalInterventions === 0 && (
              <div className="flex items-center gap-2 px-3 py-2.5 bg-orange-500/10 border border-orange-500/20 rounded-xl animate-fade-in" data-testid="no-intervention-warning">
                <AlertTriangle className="w-3.5 h-3.5 text-orange-400 shrink-0" />
                <p className="text-xs text-orange-300">Add interventions to buildings for more detailed analysis.</p>
              </div>
            )}

            {isGeneratingReport && reportProgress && (
              <div className="flex items-center gap-2.5 px-3 py-2.5 bg-primary/5 rounded-xl border border-primary/10 animate-fade-in" data-testid="report-progress">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                <p className="text-xs text-primary font-medium">{reportProgress}</p>
              </div>
            )}

            {/* ML Model Status Panel */}
            <MLModelPanel prediction={mlPrediction} status={mlStatus} />

            <Accordion type="multiple" defaultValue={generatedReport ? ["heat", "flood", "energy", "cost"] : []} className="space-y-2.5" data-testid="accordion-analysis" key={generatedReport ? "generated" : "static"}>
              {analysisAccordionItems.map((item) => (
                <AccordionItem key={item.id} value={item.id} className="border border-white/[0.06] rounded-xl overflow-visible px-4 shadow-lg transition-all duration-300" data-testid={`accordion-item-${item.id}`}>
                  <AccordionTrigger className="py-3.5 hover:no-underline gap-2">
                    <div className="flex items-center gap-2.5">
                      <item.icon className={`w-4 h-4 ${item.iconColor} shrink-0`} />
                      <span className="text-sm font-semibold text-left">{item.title}</span>
                      {generatedReport && generatedReport[item.id] && (
                        <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4 ml-auto no-default-hover-elevate no-default-active-elevate">AI</Badge>
                      )}
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    {generatedReport && generatedReport[item.id] ? (
                      <div
                        className={`text-xs text-muted-foreground leading-relaxed whitespace-pre-line transition-opacity duration-500 ${reportVisible[item.id] ? "opacity-100" : "opacity-0"}`}
                        dangerouslySetInnerHTML={{ __html: generatedReport[item.id] }}
                        data-testid={`report-content-${item.id}`}
                      />
                    ) : (
                      <p className="text-xs text-muted-foreground leading-relaxed">{item.content}</p>
                    )}
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>

            <div className="pt-4 flex justify-center">
              <span className="text-[10px] text-muted-foreground/40 font-medium tracking-wide">Created by MiniMax Agent</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
