modelData = []
let currentYear = 2023
let columnNames = ["3Month", "10Yr", "Baa", "Gold", "REIT", "SP500"]

/* Read in Model Data */

async function fetchAndDecodeMsgPackFile(filePath) {
    try {
        const response = await fetch(filePath);
        const arrayBuffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        const decodedData = msgpack.decode(uint8Array);

        // Assuming decodedData is an array of scenarios at the top level
        if (Array.isArray(decodedData)) {
            for (const scenario of decodedData) {
                if (
                    typeof scenario === "object" &&
                    scenario !== null
                ) {
                    modelData.push(scenario);
                } else {
                    console.error(`Invalid scenario structure within the file:`, scenario);
                }
            }
        } else {
            console.error(`Invalid top-level structure in ${filePath}. Expected an array of scenarios.`);
        }
    } catch (error) {
        console.error(`Error fetching or decoding ${filePath}:`, error);
    }
}

/* Plot Scenarios */
function setAlpha(rgbaColor, newAlpha) {
    return chroma(rgbaColor).alpha(newAlpha).css();
  }

  function getColor(index, numSeries) {
    const palette = new chroma.scale(['#ff8000', '#00ff00']).colors(numSeries);
    return palette[index];
  }

function plotMultipleArrays(scenarioData, years, scenario, scenarioNum) {
    let k = Object.keys(scenarioData)[0];
    if (years < 0 || years > scenarioData[k].length) {
        years = scenarioData[k].length;
    }

    let currentIndex = 0;
    let datasets = Object.keys(scenarioData).map(key => ({
      label: key,
      data: scenarioData[key].slice(0, years + 1),
      backgroundColor: getColor(currentIndex, Object.keys(scenarioData).length),
      borderColor: getColor(currentIndex++, Object.keys(scenarioData).length)
    }));

    // Add this scenario
    if (scenarioNum != 0) {
        datasets.push({
            label: "Scenario " + scenarioNum,
            data: scenario.slice(0, years + 1),
            backgroundColor: "#0000ff",
            borderColor: '#0000ff'
        });
    }
  
    const ctx = document.getElementById('myChart').getContext('2d');
  
    if (myChart != null) {
        myChart.data.labels = Array.from({length: years + 1}, (_, i) => i + currentYear);
        myChart.data.datasets = datasets;
        myChart.update();
    } else {
        myChart = new Chart(ctx, {
            type: 'line',
            data: {
              labels: Array.from({ length: years + 1 }, (_, i) => i + currentYear), // X-axis labels
              datasets: datasets 
            },
            backgroundColor: 'rgba(0,0,0,0.1)',
            options: {
              animation: {
                duration: 0
              },
              scales: {
                  y: {
                      type: isLogScale ? 'logarithmic' : 'linear'
                  }
              },
              plugins: {
                  legend: {
                      display: false
                  }
              },
              
              events: ['mousemove', 'mouseout'], // Listen for hover and mouseout
              hover: {   
                animation: {
                    duration: 10
                  },     
                  mode: 'nearest', // Or the mode you want to use        
                  intersect: true    },
              onHover: (context, event) => {
                  const activePoint = event;
                  if (activePoint.length > 0) {
                      // User is hovering over a point
                      const datasetIndex = activePoint[0].datasetIndex; 
      
                      context.chart.data.datasets.forEach((dataset, index) => {
                        
                        if (index !== datasetIndex) {
                            // Dim all datasets except the active one
                            var color = setAlpha(dataset.backgroundColor, 0.2)
                            dataset.backgroundColor = color;
                            dataset.borderColor = color;
                        } else {
                            var color = setAlpha(dataset.backgroundColor, 1)
                            dataset.backgroundColor = color;
                            dataset.borderColor = color;
                        }
                      });
                      context.chart.update();
      
                  } else {
                  // No point hovered, reset all datasets
                  context.chart.data.datasets.forEach((dataset) => {
                      var color = setAlpha(dataset.backgroundColor, 1)
                      dataset.backgroundColor = color;
                      dataset.borderColor = color;
                  });
                  context.chart.update(); 
                  }
              }
            }
        });
    }
  }

/* Calculate portfolio returns given allocation and scenarios */
function calculatePortfolioReturn(assetReturns, assetAllocation) {
    if (assetReturns.length !== assetAllocation.length) {
        return "Error: Arrays must have the same length.";
    }

    let totalReturn = 0;

    for (let i = 0; i < assetReturns.length; i++) {
        totalReturn += assetReturns[i] * assetAllocation[i];
    }

    return totalReturn;
}

function extractNthElements(jsonObject, n) {
    const resultArray = [];

    for (const key of columnNames) {
        resultArray.push(jsonObject[key][n]); 
    }

    return resultArray;
}

let startAmountElement = document.getElementById("StartAmountInput");
function calculateCumulativeReturns(scenarioData, assetAllocationList, yearList, inflowList) {
    var res = {};
    let scenarioNumber = 0;

    for (const scenario of scenarioData) {
        const portfolioValues = [startAmountElement.value];
        let allocationIndex = 0;
        let yearCount = 0;

        for (const yearsForAllocation of yearList) {
            const assetAllocation = assetAllocationList[allocationIndex];
            for (let i = 0; i < yearsForAllocation; i++) {
                // Check that scenario data is not exceeded
                if (yearCount >= Object.values(scenario)[0].length) 
                    throw "Scenario Data Exceeded";
                const assetReturns = extractNthElements(scenario, yearCount);
                const stepReturn = calculatePortfolioReturn(assetReturns, assetAllocation);

                // Calculate value at the end of this step
                let newValue = portfolioValues[portfolioValues.length-1] * (1 + stepReturn);
                newValue += inflowList[allocationIndex];
                if (newValue < 0) {
                    newValue = 0;
                }
                portfolioValues.push(newValue);
                yearCount++;
            }
            allocationIndex++;
        }

        res[`Scenario ${scenarioNumber + 1}`] = portfolioValues;
        scenarioNumber++;
    }

    return res;
}

function calculateTotalYears(yearList) {
    let totalYears = 0;
    for (let i = 0; i < yearList.length; i++) {
      totalYears += yearList[i];
    }
    return totalYears;
  }
  
  function calculatePercentiles(jsonData) {
    const arrayLength = jsonData[Object.keys(jsonData)[0]].length; // Get length of the first array, assuming all have the same length
    const percentilesToCalculate = [5, 10, 15, 25, 35, 50, 65, 75, 85, 90, 95];
    const result = {}; 
    
    percentilesToCalculate.forEach(p => {
        result[p === 0 ? "Min" : p === 100 ? "Max" : `${p}thPercentile`] = [];
    });

    for (let i = 0; i < arrayLength; i++) {
        const valuesAtTimestep = Object.values(jsonData).map(arr => arr[i]); // Get all values at the current timestep
        valuesAtTimestep.sort((a, b) => a - b); // Sort numerically

        percentilesToCalculate.forEach(percentile => {
            const index = Math.ceil(percentile / 100 * (valuesAtTimestep.length - 1)); 
            result[percentile === 0 ? "Min" : percentile === 100 ? "Max" : `${percentile}thPercentile`].push(valuesAtTimestep[index]);
        });
    }

    return result;
}

let scenarioNumElement = document.getElementById("ScenarioNum");
function UpdateModelData(allocationList, yearList, inflowList) {
    const returnsOverTime = calculateCumulativeReturns(modelData, allocationList, yearList, inflowList);
    const totalYears = yearList.reduce((sum, year) => sum + year, 0);
    var percentiles = calculatePercentiles(returnsOverTime);
    console.log(returnsOverTime['Scenario 1'])
    console.log(returnsOverTime)
    
    // Check for if a specific scenario should be chosen
    let scenario = []
    let scenarioNum = parseInt(scenarioNumElement.value);
    if (scenarioNum != 0) {
        scenario = returnsOverTime['Scenario ' + scenarioNum];
    }

    plotMultipleArrays(percentiles, totalYears, scenario, scenarioNum)
}

var ctx = document.getElementById('myChart').getContext('2d');
var isLogScale = false;  // Flag to track the scale
var toggleButton = document.getElementById('toggleScale');
var myChart = null;

toggleButton.addEventListener('click', () => {
    isLogScale = !isLogScale; 
    myChart.options.scales.y.type = isLogScale ? 'logarithmic' : 'linear';
    myChart.update(); // Redraw the chart
});

/** Allocation Logic */
var modelhidetxt = document.getElementById('badallocmsg')
var modelupdatebtn = document.getElementById('modelupdatebtn')
var allocContainer = document.getElementById('allocContainer')
modelupdatebtn.addEventListener('click', () => {
    // Get allocation 
    let children = allocContainer.children;
    let assetAllocation = [];
    let assetYears = [];
    let assetInflows = [];

    for (let i = 0; i < children.length; i++) {
        let inputs = children[i].children;
        var thisAlloc = [];
        for (let i = 1; i < 7; i++) {
            thisAlloc.push(parseInt(inputs[i].children[1].value)/100);
        }
        assetAllocation.push(thisAlloc);
        assetYears.push(parseInt(inputs[7].children[1].value));
        assetInflows.push(parseInt(inputs[8].children[1].value));
    }

    UpdateModelData(assetAllocation, assetYears, assetInflows);
})

function allocIsValid(inputElements) {
    let sum = 0;
    for (let i = 1; i < 7; i++) {
      // Handle potential non-numeric values
      const value = parseFloat(inputElements[i].children[1].value); 
      if (isNaN(value)) {
        sum = NaN;
        break;  
      } else {
        sum += value;
      }
    }
  
    // Toggle visibility based on the sum
    return (sum === 100);
}

function applyValidAllocCheck(element) {
    element.addEventListener('input', (event) => {
        // Determine if this alloc block is valid
        var parent = event.target.parentElement.parentElement;
        var isValid = allocIsValid(parent.children)
        parent.setAttribute('allocvalid', isValid);
        
        // If not valid, hide update button
        if (!isValid) {
            modelhidetxt.classList.remove('hidden');
            modelupdatebtn.classList.add('hidden');
        }
        // If this is valid, need to tell if all others are valid
        else {
            var children = parent.parentElement.children;
            for (const allocBlock of children) {
                if (!(allocBlock.getAttribute('allocvalid') == "true")) {
                    return; // Any invalid block makes entire allocation invalid
                }
            }
            // All allocations are valid
            modelhidetxt.classList.add('hidden');
            modelupdatebtn.classList.remove('hidden');
        }
    });
}

// Allocation Block Templates
var allocTemplate = document.getElementById('allocationTemplate')
function AddAllocationTemplate() {
    const clonedTemplate = allocTemplate.cloneNode(true);
    //clonedTemplate.style.display = 'block';
    clonedTemplate.classList.remove('hidden');
    for (let i = 1; i <= 6; i++) {
        var e = clonedTemplate.children[i].children[1];
        e.value = 0;
        applyValidAllocCheck(e);
    }
    clonedTemplate.removeAttribute('id');
    clonedTemplate.children[0].children[1].addEventListener('click', () => {
        allocContainer.removeChild(clonedTemplate);
        allocContainer.children[0].children[6].children[1].dispatchEvent(new Event('input'));
    })
    allocContainer.appendChild(clonedTemplate);
    clonedTemplate.children[1].children[1].dispatchEvent(new Event('input'));
}

addTemplateBtn = document.getElementById('addTemplateBtn');
addTemplateBtn.addEventListener('click', () => {
    AddAllocationTemplate();
})

AddAllocationTemplate();
allocContainer.children[0].children[6].children[1].value = 100;
allocContainer.children[0].children[7].children[1].value = 40;
allocContainer.children[0].children[6].children[1].dispatchEvent(new Event('input'));

scenarioNumElement.addEventListener('input', function() {
    if (modelupdatebtn.classList.contains('hidden')) {
        return;
    }
    modelupdatebtn.dispatchEvent(new Event('click'));
})

// Run Upon Loading

async function func() {
    console.log("Hello");
    await fetchAndDecodeMsgPackFile('https://cdn.jsdelivr.net/gh/nitzryan/PortfolioSim@master/models/scenarios.msgpack')
    modelupdatebtn.dispatchEvent(new Event('click'));
}

func()