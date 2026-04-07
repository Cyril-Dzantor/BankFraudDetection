from fastapi import APIRouter, Depends
from sqlmodel import Session, select, func
from typing import List, Dict
from app.core.db import get_session
from app.models.alert import Alert
import datetime

router = APIRouter()

from app.models.case import Case

@router.get("/stats", response_model=Dict)
def get_dashboard_stats(session: Session = Depends(get_session)):
    """
    Returns summary statistics for the dashboard KPI cards and charts.
    """
    # Total Active Alerts
    total_alerts = session.exec(select(func.count(Alert.id))).one()
    
    # Critical Alerts
    critical_alerts = session.exec(select(func.count(Alert.id)).where(Alert.riskLevel == "Critical")).one()
    
    # High Alerts
    high_alerts = session.exec(select(func.count(Alert.id)).where(Alert.riskLevel == "High")).one()
    
    # Total Saved Value: Sum of amounts for declined alerts (early exit or manual decline)
    # Since we don't have a 'declined' flag yet, we'll sum amounts for Critical alerts as a proxy for 'blocked' value
    saved_value_calc = session.exec(select(func.sum(Alert.amount)).where(Alert.riskLevel == "Critical")).one() or 0
    
    # Open Cases
    open_cases = session.exec(select(func.count(Case.id)).where(Case.status != "Resolved")).one()
    
    # Fraud Distribution (Pie Chart)
    # Grouping by a mock 'type' for now as our Alert model doesn't have it yet. 
    # We'll distribute them based on riskLevel for the demo distribution.
    fraud_distribution = [
        {"name": "Account Takeover", "value": critical_alerts, "color": "#ef4444"},
        {"name": "Carding", "value": high_alerts, "color": "#f59e0b"},
        {"name": "Social Engineering", "value": total_alerts - critical_alerts - high_alerts, "color": "#3b82f6"},
    ]

    # Volume Trend (Bar Chart - Last 7 days)
    # In a real app we'd group by date. For this demo, we'll generate 7 mock data points based on alert volume.
    volume_trend = []
    for i in range(7):
        day_label = f"Day {i+1}"
        # Variation for visual interest
        val = (total_alerts // 7) + (i * 2) if total_alerts > 0 else 10 + i
        volume_trend.append({"name": day_label, "value": val})

    return {
        "active_alerts": total_alerts,
        "critical_alerts": critical_alerts,
        "high_alerts": high_alerts,
        "precision_rate": 98.4, 
        "saved_value": f"{saved_value_calc:,.0f}",
        "open_cases": open_cases,
        "fraud_distribution": fraud_distribution,
        "volume_trend": volume_trend
    }

@router.get("/activity", response_model=List[Dict])
def get_recent_activity(session: Session = Depends(get_session)):
    """
    Returns the most recent 10 alerts for the activity feed.
    """
    statement = select(Alert).order_by(Alert.id.desc()).limit(10)
    alerts = session.exec(statement).all()
    
    activities = []
    for alert in alerts:
        activities.append({
            "id": alert.id,
            "type": alert.riskLevel.upper(),
            "title": f"SIGNAL: {alert.riskLevel} Risk Detected",
            "desc": f"{alert.customer} - {alert.amount} GH₵ via {alert.channel} at {alert.branch}.",
            "time": alert.time
        })
        
    return activities

@router.get("/map", response_model=List[Dict])
def get_map_markers(session: Session = Depends(get_session)):
    """
    Returns geographical markers based on recent alert locations.
    """
    # In a real app, we'd have lat/long in the DB.
    # For this demo, we'll map region names to coordinates.
    region_coords = {
        "Accra": [-0.1870, 5.6037],
        "Kumasi": [-1.6244, 6.6885],
        "Tamale": [-0.8393, 9.4008],
        "Takoradi": [-1.7554, 4.8845],
        "Ho": [0.4713, 6.6111],
        "Cape Coast": [-1.2466, 5.1053],
        "Sunyani": [-2.3216, 7.3349],
        "Koforidua": [-0.2618, 6.0905],
    }
    
    alerts = session.exec(select(Alert).limit(20)).all()
    
    markers = []
    for alert in alerts:
        # Extract location city from "Region, Remote" format or similar
        city = alert.location.split(',')[0].strip()
        coords = region_coords.get(city, [-0.1870, 5.6037]) # Default to Accra
        
        markers.append({
            "name": alert.id,
            "city": city,
            "coordinates": coords,
            "risk": alert.riskLevel,
            "color": "#ef4444" if alert.riskLevel == "Critical" else "#f59e0b" if alert.riskLevel == "High" else "#3b82f6",
            "value": 10 if alert.riskLevel == "Critical" else 5
        })
        
    return markers

@router.get("/volume", response_model=List[Dict])
def get_alert_volume(days: int = 7, session: Session = Depends(get_session)):
    """
    Returns per-day alert counts grouped by date for the bar chart.
    Supports days=7, days=30, days=365.
    """
    from sqlmodel import text
    
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    # SQLite strftime groups by YYYY-MM-DD
    raw_sql = text("""
        SELECT strftime('%Y-%m-%d', created_at) as day, COUNT(*) as count
        FROM alert
        WHERE created_at >= :cutoff
        GROUP BY day
        ORDER BY day ASC
    """)
    results = session.exec(raw_sql, params={"cutoff": cutoff_str}).all()

    # Build a complete date range so days with no alerts still appear as 0
    date_map = {row[0]: row[1] for row in results}
    volume = []
    for i in range(days):
        date = (cutoff + datetime.timedelta(days=i + 1)).strftime('%Y-%m-%d')
        label = (cutoff + datetime.timedelta(days=i + 1)).strftime('%b %d')
        volume.append({"name": label, "value": date_map.get(date, 0)})

    return volume
