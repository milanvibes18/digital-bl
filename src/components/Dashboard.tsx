import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { KPICard } from './KPICard'
import { DeviceCard } from './DeviceCard'
import { AlertCard } from './AlertCard'
import { 
  Activity, 
  Cpu, 
  Zap, 
  TrendingUp, 
  Wifi, 
  RefreshCw,
  Download,
  FileText,
  Bell
} from 'lucide-react'
import { Device, Alert, DashboardData, TimeRange } from '../types/digital-twin'
import { 
  getDevices, 
  getAlerts, 
  generateSampleData, 
  initializeDatabase,
  acknowledgeAlert
} from '../utils/db'
import { blink } from '../blink/client'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { cn } from '../utils/cn'

export function Dashboard() {
  const [devices, setDevices] = useState<Device[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('24h')
  const [loading, setLoading] = useState(true)
  const [userId] = useState('demo-user')

  const timeRangeOptions: { label: string; value: TimeRange }[] = [
    { label: '1H', value: '1h' },
    { label: '4H', value: '4h' },
    { label: '24H', value: '24h' },
    { label: '7D', value: '7d' },
    { label: '30D', value: '30d' }
  ]

  useEffect(() => {
    initializeApp()
  }, [])

  const initializeApp = async () => {
    try {
      setLoading(true)
      await initializeDatabase()
      
      // Check if we have any devices, if not generate sample data
      const existingDevices = await getDevices(userId)
      if (existingDevices.length === 0) {
        await generateSampleData(userId)
      }
      
      await loadData()
    } catch (error) {
      console.error('Error initializing app:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadData = async () => {
    try {
      const [devicesData, alertsData] = await Promise.all([
        getDevices(userId),
        getAlerts(userId, 10)
      ])
      
      setDevices(devicesData)
      setAlerts(alertsData)
      
      // Calculate dashboard metrics
      const totalDevices = devicesData.length
      const activeDevices = devicesData.filter(d => d.status !== 'offline').length
      const systemHealth = devicesData.reduce((avg, device) => avg + device.healthScore, 0) / totalDevices * 100
      const efficiency = devicesData.reduce((avg, device) => avg + device.efficiencyScore, 0) / totalDevices * 100
      const energyUsage = devicesData
        .filter(d => d.type === 'power_meter')
        .reduce((sum, device) => sum + device.value, 0)
      
      // Generate sample performance data
      const performanceData = Array.from({ length: 24 }, (_, i) => ({
        timestamp: `${String(i).padStart(2, '0')}:00`,
        systemHealth: systemHealth + Math.sin(i * 0.2) * 10 + Math.random() * 5,
        efficiency: efficiency + Math.cos(i * 0.15) * 8 + Math.random() * 4,
        energyUsage: energyUsage + Math.sin(i * 0.1) * energyUsage * 0.2
      }))
      
      setDashboardData({
        systemHealth: Math.round(systemHealth),
        activeDevices,
        totalDevices,
        efficiency: Math.round(efficiency),
        energyUsage: Math.round(energyUsage),
        energyCost: Math.round(energyUsage * 0.12), // $0.12 per kWh
        performanceData,
        statusDistribution: {
          normal: devicesData.filter(d => d.status === 'normal').length,
          warning: devicesData.filter(d => d.status === 'warning').length,
          critical: devicesData.filter(d => d.status === 'critical').length,
          offline: devicesData.filter(d => d.status === 'offline').length
        }
      })
    } catch (error) {
      console.error('Error loading data:', error)
    }
  }

  const handleRefresh = () => {
    loadData()
  }

  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId)
      setAlerts(alerts.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      ))
    } catch (error) {
      console.error('Error acknowledging alert:', error)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <RefreshCw className="h-6 w-6 animate-spin" />
          <span>Loading Dashboard...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-3">
                <Activity className="h-8 w-8 text-primary" />
                Digital Twin Dashboard
              </h1>
              <p className="text-muted-foreground mt-1">
                Real-time Industrial IoT Monitoring & Analytics
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <Wifi className="h-4 w-4 text-green-500" />
                <span className="text-sm text-muted-foreground">Connected</span>
              </div>
              
              <div className="flex items-center gap-2">
                <button
                  onClick={handleRefresh}
                  className="px-3 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 transition-colors flex items-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  Refresh
                </button>
                
                <button className="px-3 py-2 bg-secondary text-secondary-foreground rounded-md hover:bg-secondary/80 transition-colors flex items-center gap-2">
                  <Download className="h-4 w-4" />
                  Export
                </button>
                
                <button className="px-3 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/80 transition-colors flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  Report
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Time Range Selector */}
        <div className="mb-8">
          <div className="flex items-center gap-2">
            {timeRangeOptions.map(option => (
              <button
                key={option.value}
                onClick={() => setTimeRange(option.value)}
                className={cn(
                  "px-3 py-1 rounded-md text-sm font-medium transition-colors",
                  timeRange === option.value 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
          <KPICard
            title="System Health"
            value={`${dashboardData?.systemHealth || 0}%`}
            icon={Activity}
            variant="success"
            trend={{ value: "+2.1%", direction: "up" }}
            loading={!dashboardData}
          />
          
          <KPICard
            title="Active Devices"
            value={`${dashboardData?.activeDevices || 0}/${dashboardData?.totalDevices || 0}`}
            subtitle={`${dashboardData?.statusDistribution.offline || 0} offline`}
            icon={Cpu}
            variant="default"
            loading={!dashboardData}
          />
          
          <KPICard
            title="Energy Usage"
            value={`${dashboardData?.energyUsage || 0} kW`}
            subtitle={`$${dashboardData?.energyCost || 0}/hour`}
            icon={Zap}
            variant="warning"
            trend={{ value: "-0.8%", direction: "down" }}
            loading={!dashboardData}
          />
          
          <KPICard
            title="Efficiency"
            value={`${dashboardData?.efficiency || 0}%`}
            subtitle="Predicted 24h"
            icon={TrendingUp}
            variant="info"
            trend={{ value: "+1.2%", direction: "up" }}
            loading={!dashboardData}
          />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Performance Chart */}
          <div className="xl:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>Real-time system performance over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={dashboardData?.performanceData || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="timestamp" stroke="hsl(var(--muted-foreground))" />
                      <YAxis stroke="hsl(var(--muted-foreground))" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'hsl(var(--card))', 
                          border: '1px solid hsl(var(--border))',
                          borderRadius: '8px'
                        }} 
                      />
                      <Line 
                        type="monotone" 
                        dataKey="systemHealth" 
                        stroke="hsl(var(--success))" 
                        strokeWidth={2}
                        dot={false}
                        name="System Health %"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="efficiency" 
                        stroke="hsl(var(--primary))" 
                        strokeWidth={2}
                        dot={false}
                        name="Efficiency %"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Critical Alerts */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className="h-5 w-5" />
                  Critical Alerts
                  <span className="ml-auto bg-red-500 text-white text-xs px-2 py-1 rounded-full">
                    {alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length}
                  </span>
                </CardTitle>
                <CardDescription>Requires immediate attention</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {alerts.filter(a => a.severity === 'critical').slice(0, 3).map(alert => (
                  <AlertCard
                    key={alert.id}
                    alert={alert}
                    onAcknowledge={() => handleAcknowledgeAlert(alert.id)}
                  />
                ))}
                {alerts.filter(a => a.severity === 'critical').length === 0 && (
                  <p className="text-muted-foreground text-center py-8">
                    No critical alerts
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Devices Grid */}
        <div className="mt-8">
          <Card>
            <CardHeader>
              <CardTitle>Device Status</CardTitle>
              <CardDescription>Real-time monitoring of all connected devices</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-4">
                {devices.map(device => (
                  <DeviceCard
                    key={device.id}
                    device={device}
                    onClick={() => console.log('Device clicked:', device.id)}
                  />
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}