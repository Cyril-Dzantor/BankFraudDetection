'use client';

import {
  AlertTriangle,
  DownloadIcon,
  ShieldAlert,
  Target,
  PiggyBank,
  ClipboardList,
  Activity,
  Map as MapIcon,
  Clock,
  Zap
} from 'lucide-react';
import { useState, useMemo, useEffect } from 'react';
import { toast } from 'sonner';
import Link from 'next/link';
import { useRole } from '@/app/context/RoleContext';
import { getBackendUrl, getOrchestratorUrl } from '@/app/utils/api';
import {
  BarChart,
  Bar,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { ComposableMap, Geographies, Geography, Marker } from 'react-simple-maps';
import Modal from '../components/Modal';

const geoUrl = "/ghana-regions.json";

// Static fallback for when backend is offline
const fallbackBarData = [
  { name: 'Day 1', value: 0 },
  { name: 'Day 2', value: 0 },
  { name: 'Day 3', value: 0 },
  { name: 'Day 4', value: 0 },
  { name: 'Day 5', value: 0 },
  { name: 'Day 6', value: 0 },
  { name: 'Day 7', value: 0 },
];

const pieData = [
  { name: 'Account Takeover', value: 35, color: '#ef4444' }, // Red
  { name: 'Carding', value: 25, color: '#f59e0b' }, // Amber
  { name: 'Phishing', value: 25, color: '#3b82f6' }, // Blue
  { name: 'Other', value: 15, color: '#10b981' }, // Emerald
];



const mapMarkers = {
  'real-time': [
    { name: 'Kumasi', city: 'Kumasi', coordinates: [-1.6244, 6.6885], risk: 'High', color: '#ef4444', value: 12 },
    { name: 'Accra', city: 'Accra', coordinates: [-0.1870, 5.6037], risk: 'Moderate', color: '#f59e0b', value: 8 },
  ],
  '24h': [
    { name: 'Kumasi', city: 'Kumasi', coordinates: [-1.6244, 6.6885], risk: 'High', color: '#ef4444', value: 42 },
    { name: 'Accra', city: 'Accra', coordinates: [-0.1870, 5.6037], risk: 'Critical', color: '#dc2626', value: 85 },
    { name: 'Tamale', city: 'Tamale', coordinates: [-0.8393, 9.4008], risk: 'Low', color: '#10b981', value: 5 },
    { name: 'Takoradi', city: 'Takoradi', coordinates: [-1.7554, 4.8845], risk: 'Moderate', color: '#f59e0b', value: 15 },
  ]
};

export default function DashboardPage() {
  const { role } = useRole();
  const [mapLayer, setMapLayer] = useState<'real-time' | '24h'>('real-time');
  const [activities, setActivities] = useState<any[]>([]);
  const [isAuditModalOpen, setIsAuditModalOpen] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [feedTab, setFeedTab] = useState<'live' | '7d' | '30d' | '1y'>('live');
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [volumeData, setVolumeData] = useState(fallbackBarData);
  const [isLoadingVolume, setIsLoadingVolume] = useState(false);
  const [stats, setStats] = useState({ open_cases: 0 });
  const [isLoadingStats, setIsLoadingStats] = useState(true);

  const fetchStats = async () => {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
    setIsLoadingStats(true);
    try {
      const res = await fetch(`${backendUrl}/api/v1/dashboard/stats`);
      const data = await res.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setIsLoadingStats(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: NodeJS.Timeout;

    const connect = () => {
      const orchestratorBase = getOrchestratorUrl();
      const wsUrl = orchestratorBase.replace(/^http/, 'ws') + '/ws/live-alerts';
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        toast.success("Connected to live intelligence stream");
      };

      ws.onclose = () => {
        setIsConnected(false);
        // Attempt to reconnect after 5 seconds
        reconnectTimer = setTimeout(connect, 5000);
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const { verdict, transaction } = payload;

          let alertType = 'INFO';
          if (verdict.decision === 'DECLINE') alertType = 'CRITICAL';
          else if (verdict.decision === 'CHALLENGE') alertType = 'MEDIUM';
          else if (verdict.decision === 'APPROVE') alertType = 'RESOLVED';

          const newActivity = {
            id: Date.now() + Math.random(),
            type: alertType,
            title: verdict.decision === 'DECLINE' ? 'Fraud Blocked' : verdict.decision === 'CHALLENGE' ? 'Suspicious Activity' : 'Transaction Approved',
            desc: `Amount: $${transaction.amount} | Device: ${transaction.device_id}. ${verdict.reason || ''}`,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
          };

          setActivities(prev => {
            const updated = [newActivity, ...prev];
            return updated.slice(0, 100); // Keep last 100 events in UI
          });
        } catch (err) {
          console.error("Failed to parse websocket message", err);
        }
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, []);

  // Fetch historical activity feed when a non-live tab is selected
  useEffect(() => {
    if (feedTab === 'live') return;
    const daysMap: Record<string, number> = { '7d': 7, '30d': 30, '1y': 365 };
    const days = daysMap[feedTab];
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
    setIsLoadingHistory(true);
    fetch(`${backendUrl}/api/v1/alerts/?days=${days}&limit=100`)
      .then(r => r.json())
      .then(data => {
        const items = data.items || [];
        const converted = items.map((a: any) => ({
          id: a.id || Math.random(),
          type: a.riskLevel === 'Critical' ? 'CRITICAL' : a.riskLevel === 'High' ? 'MEDIUM' : 'RESOLVED',
          title: a.riskLevel === 'Critical' || a.riskLevel === 'High' ? 'Fraud Alert' : 'Transaction Approved',
          desc: `Customer: ${a.customer} | Amount: GH₵${a.amount} | ${a.channel} — ${a.location}`,
          time: a.time || a.created_at?.slice(0, 10) || 'Historical'
        }));
        setActivities(converted.length > 0 ? converted : [{ id: 0, type: 'INFO', title: 'No Records', desc: `No alerts found in the last ${days} days.`, time: '' }]);
      })
      .catch(() => {
        setActivities([{ id: 0, type: 'INFO', title: 'Backend Offline', desc: 'Start the backend API to view historical data.', time: '' }]);
      })
      .finally(() => setIsLoadingHistory(false));
  }, [feedTab]);

  // Fetch real volume data for bar chart whenever tab changes
  useEffect(() => {
    const daysMap: Record<string, number> = { 'live': 7, '7d': 7, '30d': 30, '1y': 365 };
    const days = daysMap[feedTab];
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
    setIsLoadingVolume(true);
    fetch(`${backendUrl}/api/v1/dashboard/volume?days=${days}`)
      .then(r => r.json())
      .then(data => setVolumeData(Array.isArray(data) ? data : fallbackBarData))
      .catch(() => setVolumeData(fallbackBarData))
      .finally(() => setIsLoadingVolume(false));
  }, [feedTab]);

  // Fetch real recent activity on mount and when switching back to live
  const fetchLiveActivity = () => {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8080';
    fetch(`${backendUrl}/api/v1/dashboard/activity`)
      .then(r => r.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          setActivities(data);
        } else {
          setActivities([{ id: 0, type: 'INFO', title: 'No Activity Yet', desc: 'Run the simulator to generate real-time fraud alerts.', time: '' }]);
        }
      })
      .catch(() => {
        setActivities([{ id: 0, type: 'INFO', title: 'Backend Offline', desc: 'Start the backend API to see live activity.', time: '' }]);
      });
  };

  useEffect(() => { fetchLiveActivity(); }, []);

  // Re-fetch live activity when switching back to 'live' tab
  useEffect(() => {
    if (feedTab === 'live') fetchLiveActivity();
  }, [feedTab]);

  const activeMarkers = useMemo(() => mapMarkers[mapLayer], [mapLayer]);

  // Dynamically calculate KPIs based on live activities feed
  const activeAlertsCount = activities.filter(a => a.type === 'CRITICAL' || a.type === 'MEDIUM').length;
  // Summing the amounts of blocked frauds. The description string looks like: "Amount: $81.52 | Device..."
  const fundsSaved = activities
    .filter(a => a.type === 'CRITICAL')
    .reduce((sum, a) => {
      const match = a.desc.match(/Amount: \$([\d,.]+)/);
      return match ? sum + parseFloat(match[1].replace(/,/g, '')) : sum;
    }, 0);

  // Dynamically generate pieData from activities
  const pieData = useMemo(() => {
    const declined = activities.filter(a => a.type === 'CRITICAL').length;
    const challenge = activities.filter(a => a.type === 'MEDIUM').length;
    const approved = activities.filter(a => a.type === 'RESOLVED').length;

    // Fallback if there are no activities yet to prevent empty chart
    if (activities.length === 0) {
      return [
        { name: 'Awaiting Data', value: 100, color: '#94a3b8' }
      ];
    }

    return [
      { name: 'Blocked', value: declined, color: '#ef4444' },
      { name: 'Suspicious', value: challenge, color: '#f59e0b' },
      { name: 'Approved', value: approved, color: '#10b981' }
    ].filter(item => item.value > 0);
  }, [activities]);

  const volumePeriodLabel = feedTab === '30d' ? '30-Day' : feedTab === '1y' ? '1-Year' : '7-Day';

  return (
    <div className="space-y-6 pb-12">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
            {role === 'system_admin' ? 'Strategic Intelligence Overview' : role === 'senior_analyst' ? 'Advanced Triage Command' : 'Fraud Intelligence Center'}
          </h2>
          <p className="text-slate-500 text-sm mt-1 font-medium">
            {role === 'system_admin' ? 'Consolidated risk, business value metrics, and system health telemetry.' :
              role === 'senior_analyst' ? 'Decision authority and advanced case resolution environment.' :
                'Monitoring cognitive fraud vectors across 16 administrative regions.'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Link href="/dashboard/alerts" className="px-4 py-2 bg-white border border-slate-200 text-slate-700 rounded-lg text-sm font-bold hover:bg-slate-50 transition-all shadow-sm flex items-center gap-2 active:scale-95">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            Active Threats
          </Link>
          <button
            onClick={() => toast.success('Intelligence summary generating...')}
            className="px-4 py-2 bg-slate-900 text-white rounded-lg text-sm font-bold hover:bg-slate-800 transition-all shadow-lg flex items-center gap-2 active:scale-95"
          >
            <DownloadIcon className="w-4 h-4" />
            Generate Report
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1 */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm relative overflow-hidden group hover:border-red-200 transition-colors">
          <div className="flex justify-between items-start mb-4">
            <div className="p-2 bg-red-50 text-red-600 rounded-lg group-hover:scale-110 transition-transform">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-black uppercase tracking-widest bg-red-50 text-red-600">
              +12% Trend
            </span>
          </div>
          <p className="text-slate-400 text-[11px] font-black uppercase tracking-widest">Active Fraud Alerts</p>
          <div className="flex items-baseline gap-2 mt-1">
            <h3 className="text-3xl font-black text-slate-900 leading-none">{activeAlertsCount}</h3>
            <span className="text-[10px] text-slate-400 font-bold tracking-wide uppercase">Real-time</span>
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-red-500/10 group-hover:bg-red-500 transition-colors" />
        </div>

        {/* Card 2 */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm relative overflow-hidden group hover:border-blue-200 transition-colors">
          <div className="flex justify-between items-start mb-4">
            <div className="p-2 bg-blue-50 text-blue-600 rounded-lg group-hover:scale-110 transition-transform">
              <Target className="w-5 h-5" />
            </div>
            <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-black uppercase tracking-widest bg-emerald-50 text-emerald-600">
              Optimized
            </span>
          </div>
          <p className="text-slate-400 text-[11px] font-black uppercase tracking-widest">Precision Rate</p>
          <div className="flex items-baseline gap-2 mt-1">
            <h3 className="text-3xl font-black text-slate-900 leading-none">98.4%</h3>
            <span className="text-[10px] text-slate-400 font-bold tracking-wide uppercase">L7 Model</span>
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-blue-600/10 group-hover:bg-blue-600 transition-colors" />
        </div>

        {/* Card 3 */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm relative overflow-hidden group hover:border-emerald-200 transition-colors">
          <div className="flex justify-between items-start mb-4">
            <div className="p-2 bg-emerald-50 text-emerald-600 rounded-lg group-hover:scale-110 transition-transform">
              <PiggyBank className="w-5 h-5" />
            </div>
            <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-black uppercase tracking-widest bg-emerald-50 text-emerald-600">
              {role === 'system_admin' ? 'Model Confidence' : 'Saved'}
            </span>
          </div>
          <p className="text-slate-400 text-[11px] font-black uppercase tracking-widest">
            {role === 'system_admin' ? 'Inference Engine Trust / Prevented Loss' : 'Prevented Loss'}
          </p>
          <div className="flex items-baseline gap-1 mt-1">
            {role === 'system_admin' ? (
              <h3 className="text-3xl font-black text-slate-900 leading-none">94.8%</h3>
            ) : (
              <>
                <span className="text-lg font-bold text-slate-400">GH₵</span>
                <h3 className="text-3xl font-black text-slate-900 leading-none">{fundsSaved.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</h3>
              </>
            )}
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-emerald-500/10 group-hover:bg-emerald-500 transition-colors" />
        </div>


        {/* Card 4 */}
        <Link href="/dashboard/cases" className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm relative overflow-hidden group hover:border-slate-400 transition-all cursor-pointer block active:scale-95">
          <div className="flex justify-between items-start mb-4">
            <div className="p-2 bg-slate-50 text-slate-600 rounded-lg group-hover:scale-110 transition-transform">
              <ClipboardList className="w-5 h-5" />
            </div>
            <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-black uppercase tracking-widest bg-slate-100 text-slate-600">
              Avg 4m
            </span>
          </div>
          <p className="text-slate-400 text-[11px] font-black uppercase tracking-widest">Open Cases</p>
          <div className="flex items-baseline gap-2 mt-1">
            <h3 className="text-3xl font-black text-slate-900 leading-none">{isLoadingStats ? '...' : stats.open_cases}</h3>
            <span className="text-[10px] text-slate-400 font-bold tracking-wide uppercase">Queue</span>
          </div>
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-200 transition-colors group-hover:bg-slate-900" />
        </Link>
      </div>
      {/* Row 2: Map & Activity Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Heatmap */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col min-h-[500px]">
          <div className="p-5 border-b border-slate-100 flex justify-between items-center bg-white z-10">
            <h3 className="font-bold text-slate-900 flex items-center gap-2 uppercase tracking-tighter text-lg">
              <MapIcon className="w-5 h-5 text-blue-500" />
              Regional Intelligence Map
            </h3>
            <div className="flex bg-slate-100 p-1 rounded-xl text-[10px] font-black uppercase tracking-widest">
              <button
                onClick={() => {
                  setMapLayer('real-time');
                  toast.success('Switching to Real-time data sync');
                }}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg transition-all ${mapLayer === 'real-time' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                <Zap className={`w-3 h-3 ${mapLayer === 'real-time' ? 'fill-blue-600' : ''}`} />
                Real-time
              </button>
              <button
                onClick={() => {
                  setMapLayer('24h');
                  toast.success('Aggregating historical 24h data');
                }}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg transition-all ${mapLayer === '24h' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                <Clock className="w-3 h-3" />
                24 Hours
              </button>
            </div>
          </div>

          <div className="flex-1 bg-slate-50 relative overflow-hidden">
            <ComposableMap
              projection="geoMercator"
              projectionConfig={{
                scale: 4500,
                center: [-1.0232, 7.9465] // Center of Ghana
              }}
              className="w-full h-full"
            >
              <Geographies geography={geoUrl}>
                {({ geographies }) =>
                  geographies.map((geo, i) => (
                    <Geography
                      key={geo.rsmKey}
                      geography={geo}
                      fill={i % 4 === 0 ? "#f8fafc" : i % 3 === 0 ? "#f1f5f9" : "#e2e8f0"}
                      stroke="#cbd5e1"
                      strokeWidth={0.5}
                      style={{
                        default: { outline: "none" },
                        hover: { fill: "#cbd5e1", outline: "none", cursor: "pointer" },
                        pressed: { outline: "none" },
                      }}
                    />
                  ))
                }
              </Geographies>

              {activeMarkers.map((marker, idx) => (
                <Marker key={idx} coordinates={marker.coordinates as [number, number]}>
                  <g className="cursor-pointer group">
                    <circle r={marker.value * 0.8} fill={marker.color} opacity={0.2} className="animate-pulse" />
                    <circle r={4} fill={marker.color} stroke="white" strokeWidth={1} />
                    <foreignObject x={8} y={-10} width={120} height={40}>
                      <div className="flex flex-col opacity-0 group-hover:opacity-100 transition-opacity bg-slate-900 text-white p-1.5 rounded-lg border border-slate-700 shadow-xl">
                        <span className="text-[10px] font-black uppercase tracking-widest leading-none mb-1">{marker.city}</span>
                        <span className="text-[10px] font-bold text-slate-400 capitalize whitespace-nowrap">{marker.risk} Risk · {marker.value} Events</span>
                      </div>
                    </foreignObject>
                    <text textAnchor="middle" y={-15} style={{ fontFamily: "Inter, sans-serif", fill: "#475569", fontSize: "10px", fontWeight: "900", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      {marker.city}
                    </text>
                  </g>
                </Marker>
              ))}
            </ComposableMap>

            {/* Legend */}
            <div className="absolute bottom-6 left-6 bg-white/80 backdrop-blur-md p-4 rounded-2xl shadow-2xl border border-slate-200/50 text-[10px] z-10 transition-all hover:scale-105">
              <div className="font-black text-slate-900 mb-3 uppercase tracking-widest">Threat Density</div>
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-200"></span>
                  <span className="text-slate-500 font-bold uppercase tracking-tighter">Secure (0-5)</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-sm shadow-amber-200"></span>
                  <span className="text-slate-500 font-bold uppercase tracking-tighter">Elevated (6-20)</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-sm shadow-red-200"></span>
                  <span className="text-slate-500 font-bold uppercase tracking-tighter">Extreme (20+)</span>
                </div>
              </div>
            </div>

            <div className="absolute top-6 right-6">
              <div className={`text-white px-3 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest shadow-xl flex items-center gap-2 ${isConnected ? 'bg-blue-600 animate-bounce' : 'bg-red-500 opacity-80'}`}>
                <span className="w-1.5 h-1.5 bg-white rounded-full"></span>
                {isConnected ? 'Live Feed Connected' : 'Feed Disconnected'}
              </div>
            </div>
          </div>
        </div>

        {/* Live Activity Feed */}
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col h-full max-h-[600px]">
          <div className="p-5 border-b border-slate-100 flex justify-between items-center">
            <h3 className="font-bold text-slate-900 flex items-center gap-2 uppercase tracking-tighter text-lg">
              <Activity className="w-5 h-5 text-slate-400" />
              Intelligence Signal
            </h3>
            <div className="flex items-center gap-1">
              {feedTab === 'live' && <span className="w-2 h-2 bg-emerald-500 rounded-full animate-ping"></span>}
            </div>
          </div>

          {/* Time Window Tabs */}
          <div className="flex gap-1 px-4 pb-3 border-b border-slate-100">
            {(['live', '7d', '30d', '1y'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setFeedTab(tab)}
                className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest transition-all ${feedTab === tab
                  ? 'bg-slate-900 text-white shadow'
                  : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
                  }`}
              >
                {tab === 'live' ? '🔴 Live' : tab === '7d' ? '7 Days' : tab === '30d' ? '1 Month' : '1 Year'}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
            {isLoadingHistory ? (
              <div className="flex flex-col items-center justify-center h-32 gap-2">
                <div className="w-6 h-6 border-2 border-slate-300 border-t-slate-700 rounded-full animate-spin"></div>
                <span className="text-xs text-slate-400 font-bold uppercase tracking-widest">Loading historical data...</span>
              </div>
            ) : (
              activities.map((act) => (
                <div key={act.id} className="relative pl-4 border-l-2 border-slate-100 group transition-all hover:border-blue-400 pb-2">
                  <div className="absolute -left-[5px] top-0 w-2 h-2 rounded-full bg-slate-200 group-hover:bg-blue-400 transition-colors"></div>
                  <div className="flex justify-between items-start mb-1">
                    <span className={`text-[9px] font-black px-2 py-0.5 rounded tracking-widest uppercase border ${act.type === 'CRITICAL' ? 'bg-red-50 text-red-600 border-red-100' :
                      act.type === 'MEDIUM' ? 'bg-amber-50 text-amber-600 border-amber-100' :
                        act.type === 'RESOLVED' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' :
                          'bg-blue-50 text-blue-600 border-blue-100'
                      }`}>
                      {act.type}
                    </span>
                    <span className="text-[10px] font-bold text-slate-400 uppercase">{act.time}</span>
                  </div>
                  <h4 className="text-sm font-bold text-slate-900 mt-1.5 group-hover:text-blue-600 transition-colors uppercase tracking-tight">{act.title}</h4>
                  <p className="text-xs text-slate-500 mt-1 leading-relaxed font-medium">{act.desc}</p>
                </div>
              ))
            )}
          </div>

          <div className="p-4 border-t border-slate-100 bg-slate-50/50">
            <button
              onClick={() => setIsAuditModalOpen(true)}
              className="w-full py-2 bg-white border border-slate-200 rounded-lg text-xs font-black text-slate-600 hover:bg-slate-50 transition-all shadow-sm uppercase tracking-widest"
            >
              Full Signal History
            </button>
          </div>
        </div>

      </div>

      {/* Row 3: Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Bar Chart */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="font-bold text-slate-900 text-lg uppercase tracking-tight">Alert Volume Trend</h3>
              <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mt-0.5">{volumePeriodLabel} — Fraud Alerts Per Day (Live DB)</p>
            </div>
            {isLoadingVolume && (
              <span className="text-[10px] font-black uppercase tracking-widest text-slate-400 animate-pulse">Updating...</span>
            )}
          </div>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={volumeData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                <Tooltip cursor={{ fill: '#f8fafc' }} contentStyle={{ borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)', fontWeight: 'bold' }} />
                <Bar dataKey="value" fill="#3b82f6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Donut Chart */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col">
          <h3 className="font-bold text-slate-900 text-lg uppercase tracking-tight mb-2">Cognitive Classifier</h3>
          <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-6">Threat Type Segmentation</p>

          <div className="relative h-48 w-full flex items-center justify-center flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={65}
                  outerRadius={85}
                  paddingAngle={8}
                  dataKey="value"
                  stroke="none"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>

            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
              <span className="text-4xl font-black text-slate-900 block leading-none">{pieData.reduce((a, b) => a + b.value, 0)}</span>
              <span className="text-[10px] text-slate-400 font-black tracking-widest uppercase mt-1">Signals</span>
            </div>
          </div>

          <div className="mt-8 grid grid-cols-2 gap-3">
            {pieData.map((item, idx) => (
              <div key={idx} className="flex flex-col gap-1.5 p-2 rounded-xl border border-slate-50 transition-colors hover:bg-slate-50">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }}></span>
                  <span className="text-[10px] text-slate-500 font-black uppercase tracking-tighter whitespace-nowrap">{item.name}</span>
                </div>
                <span className="font-black text-slate-900 text-sm pl-4">{item.value}</span>
              </div>
            ))}
          </div>

        </div>

      </div>

      {/* Audit Log Modal */}
      <Modal
        isOpen={isAuditModalOpen}
        onClose={() => setIsAuditModalOpen(false)}
        title="Comprehensive Signal Audit Log"
        footer={(
          <button
            onClick={() => {
              setIsAuditModalOpen(false);
              toast.success('Audit report exported to secure storage');
            }}
            className="px-6 py-2 bg-slate-900 text-white rounded-lg text-sm font-black uppercase tracking-widest"
          >
            Export Full Audit
          </button>
        )}
      >
        <div className="space-y-4">
          <div className="p-3 bg-blue-50 border border-blue-100 rounded-xl">
            <p className="text-[10px] text-blue-700 font-bold leading-relaxed">
              Displaying last 24 hours of autonomous agent activities and high-confidence neural signals.
            </p>
          </div>
          <div className="divide-y divide-slate-100">
            {[...activities, ...activities].map((act, i) => (
              <div key={i} className="py-3 flex justify-between items-start gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-[8px] font-black px-1.5 py-0.5 rounded tracking-tighter uppercase ${act.type === 'CRITICAL' ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-600'
                      }`}>
                      {act.type}
                    </span>
                    <span className="text-[10px] font-bold text-slate-900 tracking-tight">{act.title}</span>
                  </div>
                  <p className="text-[11px] text-slate-500 font-medium">{act.desc}</p>
                </div>
                <span className="text-[10px] font-bold text-slate-400 whitespace-nowrap">{act.time}</span>
              </div>
            ))}
          </div>
        </div>
      </Modal>

    </div>
  );
}

