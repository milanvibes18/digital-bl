import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { KPICard } from './KPICard'
import { DeviceCard } from './DeviceCard'
import { AlertCard } from './AlertCard'
import { ThresholdSettings } from './ThresholdSettings'
import { EmailLogsView } from './EmailLogsView'
import { 
  Activity, 
  Cpu, 
  Zap, 
  TrendingUp, 
  Wifi, 
  RefreshCw,
  Download,
  FileText,
  Bell,
  Settings,
  Mail
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
import { DataSimulator, defaultThresholds } from '../utils/data-simulator'
import { useToast } from '../hooks/use-toast'
import { Button } from './ui/button'

export function Dashboard() {
  const [devices, setDevices] = useState<Device[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [timeRange, setTimeRange] = useState<TimeRange>('24h')
  const [loading, setLoading] = useState(true)
  const [userId] = useState('demo-user')
  const [lastUpdate, setLastUpdate] = useState(Date.now())
  const [showThresholds, setShowThresholds] = useState(false)
  const [showEmailLogs, setShowEmailLogs] = useState(false)
  const [newCriticalAlerts, setNewCriticalAlerts] = useState<Set<string>>(new Set())
  const [isExporting, setIsExporting] = useState(false)
  
  const simulatorRef = useRef<DataSimulator | null>(null)
  const { toast } = useToast()

  const timeRangeOptions: { label: string; value: TimeRange }[] = [
    { label: '1H', value: '1h' },
    { label: '4H', value: '4h' },
    { label: '24H', value: '24h' },
    { label: '7D', value: '7d' },
    { label: '30D', value: '30d' }
  ]

  useEffect(() => {
    initializeApp()
    
    return () => {
      // Cleanup simulator on component unmount
      if (simulatorRef.current) {
        simulatorRef.current.stop()
      }
    }
  }, [])

  // Set up real-time data refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (!loading) {
        loadData()
        setLastUpdate(Date.now())
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [loading])

  const initializeApp = async () => {
    try {
      setLoading(true)
      await initializeDatabase()
      
      // Check if we have any devices, if not generate sample data
      const existingDevices = await getDevices(userId)
      if (existingDevices.length === 0) {
        await generateSampleData(userId)
      }
      
      // Initialize default thresholds if none exist
      const existingThresholds = await (blink.db as any).alertThresholds.list({ where: { userId } })
      if (existingThresholds.length === 0) {
        await Promise.all(
          defaultThresholds.map((threshold: any) => 
            (blink.db as any).alertThresholds.create({
              ...threshold,
              id: `THRESH_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              userId
            })
          )
        )
      }
      
      // Start the data simulator
      simulatorRef.current = new DataSimulator(userId)
      simulatorRef.current.start()
      
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
      
      // Check for new critical alerts and add pulse effect
      const previousCriticalAlerts = alerts.filter(a => a.severity === 'critical' && !a.acknowledged)
      const currentCriticalAlerts = alertsData.filter(a => a.severity === 'critical' && !a.acknowledged)
      
      const newAlerts = currentCriticalAlerts.filter(current => 
        !previousCriticalAlerts.some(prev => prev.id === current.id)
      )
      
      if (newAlerts.length > 0) {
        const newAlertIds = new Set(newAlerts.map(alert => alert.id))
        setNewCriticalAlerts(newAlertIds as Set<string>)
        
        // Clear the pulse effect after 3 seconds
        setTimeout(() => {
          setNewCriticalAlerts(new Set<string>())
        }, 3000)
      }
      
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
      
      // Generate sample performance data with real-time variation
      const performanceData = Array.from({ length: 24 }, (_, i) => ({
        timestamp: `${String(i).padStart(2, '0')}:00`,
        systemHealth: Math.max(0, Math.min(100, systemHealth + Math.sin(i * 0.2) * 10 + Math.random() * 5)),
        efficiency: Math.max(0, Math.min(100, efficiency + Math.cos(i * 0.15) * 8 + Math.random() * 4)),
        energyUsage: Math.max(0, energyUsage + Math.sin(i * 0.1) * energyUsage * 0.2)
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
    setLastUpdate(Date.now())
    toast({
      title: "Data Refreshed",
      description: "Dashboard data has been updated."
    })
  }

  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      await acknowledgeAlert(alertId)
      setAlerts(alerts.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      ))
      toast({
        title: "Alert Acknowledged",
        description: "The alert has been marked as acknowledged."
      })
    } catch (error) {
      console.error('Error acknowledging alert:', error)
      toast({
        title: "Error",
        description: "Failed to acknowledge alert.",
        variant: "destructive"
      })
    }
  }

  const handleExport = async () => {
    try {
      setIsExporting(true)
      
      // Generate CSV data
      const csvData = [
        ['Device ID', 'Name', 'Type', 'Status', 'Value', 'Unit', 'Health Score', 'Efficiency Score', 'Location', 'Last Updated'],
        ...devices.map(device => [
          device.id,
          device.name,
          device.type,
          device.status,
          device.value.toString(),
          device.unit,
          (device.healthScore * 100).toFixed(1) + '%',
          (device.efficiencyScore * 100).toFixed(1) + '%',
          device.location,
          new Date(device.timestamp).toLocaleString()
        ])
      ]
      
      const csvContent = csvData.map(row => row.join(',')).join('\n')
      const blob = new Blob([csvContent], { type: 'text/csv' })
      const url = window.URL.createObjectURL(blob)
      
      const a = document.createElement('a')
      a.href = url
      a.download = `iot-dashboard-export-${new Date().toISOString().split('T')[0]}.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
      
      toast({
        title: "Export Successful",
        description: "Device data has been exported to CSV."
      })
    } catch (error) {
      console.error('Error exporting data:', error)
      toast({
        title: "Export Failed",
        description: "Failed to export data.",
        variant: "destructive"
      })
    } finally {
      setIsExporting(false)
    }
  }

  const handleGenerateReport = async () => {
    try {
      setIsExporting(true)
      
      // Generate comprehensive report data
      const reportData = {
        generatedAt: new Date().toISOString(),
        summary: {
          totalDevices: devices.length,
          activeDevices: devices.filter(d => d.status !== 'offline').length,
          systemHealth: dashboardData?.systemHealth || 0,
          efficiency: dashboardData?.efficiency || 0,
          energyUsage: dashboardData?.energyUsage || 0
        },
        devices: devices.map(device => ({
          ...device,
          healthScorePercent: (device.healthScore * 100).toFixed(1),
          efficiencyScorePercent: (device.efficiencyScore * 100).toFixed(1)
        })),
        alerts: alerts.slice(0, 20), // Include recent alerts
        statusDistribution: dashboardData?.statusDistribution
      }
      
      const reportJson = JSON.stringify(reportData, null, 2)
      const blob = new Blob([reportJson], { type: 'application/json' })
      const url = window.URL.createObjectURL(blob)
      
      const a = document.createElement('a')
      a.href = url
      a.download = `iot-dashboard-report-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)
      
      toast({
        title: "Report Generated",
        description: "Comprehensive system report has been generated."
      })
    } catch (error) {
      console.error('Error generating report:', error)
      toast({
        title: "Report Generation Failed",
        description: "Failed to generate report.",
        variant: "destructive"
      })
    } finally {
      setIsExporting(false)
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
                <span className="text-xs text-muted-foreground/70 ml-2">
                  Last update: {new Date(lastUpdate).toLocaleTimeString()}
                </span>
              </div>
              
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowThresholds(true)}
                >
                  <Settings className="h-4 w-4" />
                  Settings
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowEmailLogs(true)}
                >
                  <Mail className="h-4 w-4" />
                  Email Logs
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefresh}
                >
                  <RefreshCw className="h-4 w-4" />
                  Refresh
                </Button>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExport}
                  disabled={isExporting}
                >
                  <Download className="h-4 w-4" />
                  Export
                </Button>
                
                <Button
                  size="sm"
                  onClick={handleGenerateReport}
                  disabled={isExporting}
                >
                  <FileText className="h-4 w-4" />
                  Report
                </Button>
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
            <Card className={cn(
              "transition-all duration-300",
              newCriticalAlerts.size > 0 && "ring-2 ring-red-500 shadow-red-500/20 shadow-lg animate-pulse"
            )}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bell className={cn(
                    "h-5 w-5",
                    newCriticalAlerts.size > 0 && "text-red-500 animate-pulse"
                  )} />
                  Critical Alerts
                  <span className={cn(
                    "ml-auto text-white text-xs px-2 py-1 rounded-full transition-colors",
                    newCriticalAlerts.size > 0 
                      ? "bg-red-600 animate-pulse" 
                      : "bg-red-500"
                  )}>
                    {alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length}
                  </span>
                </CardTitle>
                <CardDescription>Requires immediate attention</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {alerts.filter(a => a.severity === 'critical').slice(0, 3).map(alert => (
                  <div
                    key={alert.id}
                    className={cn(
                      "transition-all duration-300",
                      newCriticalAlerts.has(alert.id) && "ring-2 ring-red-400 shadow-md animate-pulse"
                    )}
                  >
                    <AlertCard
                      alert={alert}
                      onAcknowledge={() => handleAcknowledgeAlert(alert.id)}
                    />
                  </div>
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

      {/* Settings Modal */}
      {showThresholds && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <ThresholdSettings 
            userId={userId} 
            onClose={() => setShowThresholds(false)} 
          />
        </div>
      )}

      {/* Email Logs Modal */}
      {showEmailLogs && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <EmailLogsView 
            userId={userId} 
            onClose={() => setShowEmailLogs(false)} 
          />
        </div>
      )}
    </div>
  )
}