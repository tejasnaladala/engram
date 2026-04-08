import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Area, AreaChart } from 'recharts'

interface Props {
  history: { tick: number; error: number }[]
}

export default function PredictionError({ history }: Props) {
  const data = history.slice(-200)

  return (
    <div className="w-full h-full p-2" style={{ height: 'calc(100% - 28px)' }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="errorGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ff8c00" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ff8c00" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="tick"
            tick={{ fontSize: 9, fill: '#8888aa' }}
            tickLine={false}
            axisLine={{ stroke: 'rgba(255,255,255,0.05)' }}
          />
          <YAxis
            tick={{ fontSize: 9, fill: '#8888aa' }}
            tickLine={false}
            axisLine={{ stroke: 'rgba(255,255,255,0.05)' }}
            domain={[0, 'auto']}
          />
          <Area
            type="monotone"
            dataKey="error"
            stroke="#ff8c00"
            strokeWidth={2}
            fill="url(#errorGradient)"
            animationDuration={0}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
