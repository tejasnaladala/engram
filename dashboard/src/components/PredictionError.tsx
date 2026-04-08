import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts'

interface PredictionErrorProps {
  history: { tick: number; error: number }[]
}

export default function PredictionError({ history }: PredictionErrorProps) {
  const data = history.slice(-200)
  const cur = data.length > 0 ? data[data.length - 1].error : 0

  return (
    <div style={{ width: '100%', height: 'calc(100% - 22px)', padding: '2px 4px 2px 0', position: 'relative' }}>
      {/* Live readout */}
      <div style={{
        position: 'absolute', top: '4px', right: '8px', zIndex: 2,
        display: 'flex', alignItems: 'baseline', gap: '3px',
      }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: '16px', fontWeight: 500,
          color: '#f09030', fontVariantNumeric: 'tabular-nums',
          textShadow: '0 0 10px rgba(240,144,48,0.25)',
        }}>{cur.toFixed(3)}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '7px', color: 'var(--text-dim)', letterSpacing: '0.5px' }}>PE</span>
      </div>

      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 16, right: 4, left: -24, bottom: 0 }}>
          <defs>
            <linearGradient id="peG" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f09030" stopOpacity={0.18} />
              <stop offset="70%" stopColor="#f09030" stopOpacity={0.02} />
              <stop offset="100%" stopColor="#f09030" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="tick" tick={{ fontSize: 7, fill: '#2a3450', fontFamily: 'var(--font-mono)' }}
            tickLine={false} axisLine={{ stroke: 'rgba(80,120,180,0.06)' }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 7, fill: '#2a3450', fontFamily: 'var(--font-mono)' }}
            tickLine={false} axisLine={{ stroke: 'rgba(80,120,180,0.06)' }}
            domain={[0, 'auto']} width={30} />
          <ReferenceLine y={0} stroke="rgba(80,120,180,0.06)" strokeDasharray="2 4" />
          <Area type="monotone" dataKey="error" stroke="#f09030" strokeWidth={1.5}
            fill="url(#peG)" animationDuration={0} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
