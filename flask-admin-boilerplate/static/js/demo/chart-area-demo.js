// Set new default font family and font color to mimic Bootstrap's default styling
(Chart.defaults.global.defaultFontFamily = "Nunito"),
  '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = "#858796";

function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + "").replace(",", "").replace(" ", "");
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = typeof thousands_sep === "undefined" ? "," : thousands_sep,
    dec = typeof dec_point === "undefined" ? "." : dec_point,
    s = "",
    toFixedFix = function (n, prec) {
      var k = Math.pow(10, prec);
      return "" + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : "" + Math.round(n)).split(".");
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || "").length < prec) {
    s[1] = s[1] || "";
    s[1] += new Array(prec - s[1].length + 1).join("0");
  }
  return s.join(dec);
}

// Area Chart Example
const testing = async function () {
  const test = await fetch("/api");
  const response = await test.json();
  console.log(response["timestamp"]);
  const timestamps = [];
  const NonAggressive = [];
  const AggressiveRightTurn = [];
  const AggressiveLeftTurn = [];
  const AggressiveRightLaneChange = [];
  const AggressiveLeftLaneChange = [];
  const AggressiveAcceleration = [];
  const AggressiveBraking = [];

  for (const [key, value] of Object.entries(response["timestamp"])) {
    timestamps.push(value);
  }

  for (const [key, value] of Object.entries(response["Non-aggressive Event"])) {
    NonAggressive.push(value);
  }

  for (const [key, value] of Object.entries(
    response["Aggressive Right Turn"]
  )) {
    AggressiveRightTurn.push(value);
  }

  for (const [key, value] of Object.entries(response["Aggressive Left Turn"])) {
    AggressiveLeftTurn.push(value);
  }

  for (const [key, value] of Object.entries(
    response["Aggressive Right Lane Change"]
  )) {
    AggressiveRightLaneChange.push(value);
  }

  for (const [key, value] of Object.entries(
    response["Aggressive Left Lane Change"]
  )) {
    AggressiveLeftLaneChange.push(value);
  }

  for (const [key, value] of Object.entries(
    response["Aggressive Acceleration"]
  )) {
    AggressiveAcceleration.push(value);
  }

  for (const [key, value] of Object.entries(response["Aggressive Braking"])) {
    AggressiveBraking.push(value);
  }

  var ctx = document.getElementById("myAreaChart");
  var myLineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: timestamps,
      datasets: [
        {
          label: "Non-aggressive Event",
          lineTension: 0.3,
          backgroundColor: "rgba(78, 115, 223, 0.05)",
          borderColor: "rgba(8, 54, 42, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(8, 54, 42, 1)",
          pointBorderColor: "rgba(8, 54, 42, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(8, 54, 42, 1)",
          pointHoverBorderColor: "rgba(8, 54, 42, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: NonAggressive,
        },
        {
          label: "Aggressive Right Turn",
          lineTension: 0.3,
          backgroundColor: "rgba(78, 115, 223, 0.05)",
          borderColor: "rgba(78, 115, 223, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(78, 115, 223, 1)",
          pointBorderColor: "rgba(78, 115, 223, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
          pointHoverBorderColor: "rgba(78, 115, 223, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: AggressiveRightTurn,
        },
        {
          label: "Aggressive Left Turn",
          lineTension: 0.3,
          backgroundColor: "rgba(78, 115, 223, 0.05)",
          borderColor: "rgba(255, 99, 71, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(255, 99, 71, 1)",
          pointBorderColor: "rgba(255, 99, 71, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(255, 99, 71, 1)",
          pointHoverBorderColor: "rgba(255, 99, 71, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: AggressiveLeftTurn,
        },
        {
          label: "Aggressive Right Lane Change",
          lineTension: 0.3,
          backgroundColor: "rgba(78, 115, 223, 0.05)",
          borderColor: "rgba(73, 40, 163, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(73, 40, 163, 1)",
          pointBorderColor: "rgba(73, 40, 163, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(73, 40, 163, 1)",
          pointHoverBorderColor: "rgba(73, 40, 163, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: AggressiveRightLaneChange,
        },
        {
          label: "Aggressive Left Lane Change",
          lineTension: 0.3,
          backgroundColor: "rgba(78, 115, 223, 0.05)",
          borderColor: "rgba(128, 29, 134, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(128, 29, 134, 1)",
          pointBorderColor: "rgba(128, 29, 134, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(128, 29, 134, 1)",
          pointHoverBorderColor: "rgba(128, 29, 134, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: AggressiveLeftLaneChange,
        },
        {
          label: "Aggressive Acceleration",
          lineTension: 0.3,
          backgroundColor: "rgba(78, 115, 223, 0.05)",
          borderColor: "rgba(248, 82, 6, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(248, 82, 6, 1)",
          pointBorderColor: "rgba(248, 82, 6, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(248, 82, 6, 1)",
          pointHoverBorderColor: "rgba(248, 82, 6, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: AggressiveAcceleration,
        },
        {
          label: "Aggressive Braking",
          lineTension: 0.3,
          backgroundColor: "rgba(78, 115, 223, 0.05)",
          borderColor: "rgba(20, 180, 133, 1)",
          pointRadius: 3,
          pointBackgroundColor: "rgba(20, 180, 133, 1)",
          pointBorderColor: "rgba(20, 180, 133, 1)",
          pointHoverRadius: 3,
          pointHoverBackgroundColor: "rgba(20, 180, 133, 1)",
          pointHoverBorderColor: "rgba(20, 180, 133, 1)",
          pointHitRadius: 10,
          pointBorderWidth: 2,
          data: AggressiveBraking,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      layout: {
        padding: {
          left: 10,
          right: 25,
          top: 25,
          bottom: 0,
        },
      },
      scales: {
        xAxes: [
          {
            time: {
              unit: "date",
            },
            gridLines: {
              display: false,
              drawBorder: false,
            },
            ticks: {
              maxTicksLimit: 7,
            },
          },
        ],
        yAxes: [
          {
            ticks: {
              maxTicksLimit: 5,
              padding: 10,
              // Include a dollar sign in the ticks
              callback: function (value, index, values) {
                return value;
              },
            },
            gridLines: {
              color: "rgb(234, 236, 244)",
              zeroLineColor: "rgb(234, 236, 244)",
              drawBorder: false,
              borderDash: [2],
              zeroLineBorderDash: [2],
            },
          },
        ],
      },
      legend: {
        display: false,
      },
      tooltips: {
        backgroundColor: "rgb(255,255,255)",
        bodyFontColor: "#858796",
        titleMarginBottom: 10,
        titleFontColor: "#6e707e",
        titleFontSize: 14,
        borderColor: "#dddfeb",
        borderWidth: 1,
        xPadding: 15,
        yPadding: 15,
        displayColors: false,
        intersect: false,
        mode: "index",
        caretPadding: 10,
        callbacks: {
          label: function (tooltipItem, chart) {
            var datasetLabel =
              chart.datasets[tooltipItem.datasetIndex].label || "";
            return datasetLabel + ": " + tooltipItem.yLabel;
          },
        },
      },
    },
  });
};

testing();
