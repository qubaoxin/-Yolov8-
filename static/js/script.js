document.addEventListener('DOMContentLoaded', async () => {
  const response = await fetch('/traffic_data');
  const data = await response.json();

  const ctx = document.getElementById('trafficChart');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Car', 'Motorcycle', 'Bus', 'Truck'],
      datasets: [{
        label: '实时车辆数量',
        data: [
          data.counts_per_class?.car || 0,
          data.counts_per_class?.motorcycle || 0,
          data.counts_per_class?.bus || 0,
          data.counts_per_class?.truck || 0
        ],
        backgroundColor: ['#FF6384','#36A2EB','#4BC0C0','#FFCD56']
      }]
    },
    options: { responsive: true }
  });
});
