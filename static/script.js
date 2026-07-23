// API Config
const API_BASE = ""; // Relative path to API server on same host/port

// State
let allEntities = [];

// DOM Loaded Initialization
document.addEventListener("DOMContentLoaded", () => {
    // Set default date for bike prediction (today)
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');
    const dd = String(today.getDate()).padStart(2, '0');
    document.getElementById("bike-date").value = `${yyyy}-${mm}-${dd}`;
    
    // Initial hour label
    updateHourLabel(12);
    updateYearLabel(2026);
    
    // Fetch initial model metadata
    fetchMetadata();
});

// Switching Dashboard Tabs
function switchTab(tabName) {
    // Update button states
    document.querySelectorAll(".nav-btn").forEach(btn => btn.classList.remove("active"));
    document.getElementById(`tab-btn-${tabName}`).classList.add("active");
    
    // Update section visibility
    document.querySelectorAll(".dashboard-section").forEach(sec => sec.classList.remove("active"));
    document.getElementById(`section-${tabName}`).classList.add("active");
}

// Update Hour Slider Label
function updateHourLabel(val) {
    const formattedHour = String(val).padStart(2, '0') + ":00";
    document.getElementById("hour-val").innerText = formattedHour;
}

// Update Year Slider Label
function updateYearLabel(val) {
    document.getElementById("year-val").innerText = val;
}

// Fetch Backend Model Metadata
async function fetchMetadata() {
    try {
        const res = await fetch(`${API_BASE}/api/metadata`);
        if (!res.ok) throw new Error("Metadata fetch failed");
        
        const data = await res.json();
        
        // Update model status indicators in sidebar
        const bikeIndicator = document.getElementById("bike-status");
        if (data.bike_model_loaded) {
            bikeIndicator.classList.add("loaded");
            bikeIndicator.classList.remove("error");
        } else {
            bikeIndicator.classList.add("error");
            bikeIndicator.classList.remove("loaded");
        }

        const wellbeingIndicator = document.getElementById("wellbeing-status");
        if (data.wellbeing_model_loaded) {
            wellbeingIndicator.classList.add("loaded");
            wellbeingIndicator.classList.remove("error");
        } else {
            wellbeingIndicator.classList.add("error");
            wellbeingIndicator.classList.remove("loaded");
        }
        
        // Populate wellbeing country dropdown
        if (data.wellbeing_entities && data.wellbeing_entities.length > 0) {
            allEntities = data.wellbeing_entities;
            populateEntityDropdown(allEntities);
        }
    } catch (err) {
        console.error("Error connecting to API server:", err);
        document.getElementById("bike-status").classList.add("error");
        document.getElementById("wellbeing-status").classList.add("error");
    }
}

// Populate Country Selector Options
function populateEntityDropdown(entities) {
    const select = document.getElementById("wellbeing-entity");
    select.innerHTML = "";
    
    entities.forEach(entity => {
        const opt = document.createElement("option");
        opt.value = entity;
        opt.innerText = entity;
        select.appendChild(opt);
    });
    
    // Pre-select first item if available
    if (entities.length > 0) {
        select.selectedIndex = 0;
    }
}

// Filter Countries in Select Option list
function filterEntities(query) {
    const filtered = allEntities.filter(entity => 
        entity.toLowerCase().includes(query.toLowerCase())
    );
    populateEntityDropdown(filtered);
}

// Handle Form Submission for Bike Model
async function handleBikeSubmit(event) {
    event.preventDefault();
    
    // Show Loading state
    const submitBtn = document.getElementById("bike-submit");
    submitBtn.innerText = "Running Model Forecast...";
    submitBtn.disabled = true;
    
    // Extract values
    const rawDate = document.getElementById("bike-date").value; // yyyy-mm-dd
    let formattedDate = "";
    if (rawDate) {
        const parts = rawDate.split("-");
        formattedDate = `${parts[2]}/${parts[1]}/${parts[0]}`; // dd/mm/yyyy
    }
    
    const requestData = {
        Date: formattedDate,
        Hour: parseInt(document.getElementById("bike-hour").value),
        Temperature: parseFloat(document.getElementById("bike-temp").value),
        Humidity: parseFloat(document.getElementById("bike-humidity").value),
        WindSpeed: parseFloat(document.getElementById("bike-wind").value),
        Visibility: parseFloat(document.getElementById("bike-visibility").value),
        Season: document.getElementById("bike-season").value === "Auto" ? null : document.getElementById("bike-season").value,
        Holiday: document.getElementById("bike-holiday").value,
        FunctioningDay: document.getElementById("bike-functioning").value,
        Rainfall: parseFloat(document.getElementById("bike-rain").value),
        Snowfall: parseFloat(document.getElementById("bike-snow").value),
        SolarRadiation: parseFloat(document.getElementById("bike-solar").value)
    };
    
    try {
        const res = await fetch(`${API_BASE}/api/predict/bike`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData)
        });
        
        if (!res.ok) {
            const errDetails = await res.json();
            throw new Error(errDetails.detail || "Server returned an error");
        }
        
        const result = await res.json();
        
        // Hide Placeholder, Display Results
        document.getElementById("bike-placeholder").classList.add("hidden");
        const resPanel = document.getElementById("bike-result");
        resPanel.classList.remove("hidden");
        
        // Populate Result metrics
        const prediction = result.prediction;
        document.getElementById("bike-pred-val").innerText = prediction;
        document.getElementById("bike-explanation").innerText = result.explanation;
        
        // Update tags
        const tagsContainer = document.getElementById("bike-tags");
        tagsContainer.innerHTML = "";
        const tags = [
            `Holiday: ${requestData.Holiday}`,
            `Service Day: ${requestData.FunctioningDay}`,
            `Time: ${String(requestData.Hour).padStart(2, '0')}:00`
        ];
        tags.forEach(tagText => {
            const tagEl = document.createElement("span");
            tagEl.className = "tag";
            tagEl.innerText = tagText;
            tagsContainer.appendChild(tagEl);
        });

        // Set visual indicator fill width (capped at 2500 max bikes for percentage display)
        const percentage = Math.min(100, (prediction / 2500) * 100);
        document.getElementById("bike-indicator-fill").style.width = `${percentage}%`;
        
        // Generate contextual analysis text
        generateBikeInsights(prediction, requestData);
        
    } catch (err) {
        alert("Prediction Failed: " + err.message);
    } finally {
        submitBtn.innerText = "Run Prediction Model";
        submitBtn.disabled = false;
    }
}

// Generate Bike Insights Panel HTML
function generateBikeInsights(prediction, request) {
    const insightBox = document.getElementById("bike-insight");
    let title = "";
    let body = "";
    
    if (request.FunctioningDay === "No") {
        title = "⚠️ Service Inactive";
        body = "The bike sharing system is marked as <b>non-functioning</b>. Demand is forced to 0 by domain logic since rentals are disabled under these conditions (e.g. system maintenance or severe weather closures).";
    } else if (prediction > 1500) {
        title = "📈 Exceptionally High Demand Expected";
        body = "Environmental conditions are highly favorable (ideal temperature, high visibility, no rainfall). <b>Recommendation:</b> Deploy additional bikes to active stations and monitor logistics to prevent stock-outs.";
    } else if (prediction > 800) {
        title = "👍 Stable Demand Forecast";
        body = "Moderate bike usage predicted. Conditions indicate steady active transit levels. Standard operational schedules should suffice.";
    } else {
        title = "📉 Low Traffic Forecasted";
        body = "Predicted demand is low. This is typically driven by colder temperatures, precipitation (rain/snow), late night hours, or holidays. Restocking schedules can be optimized for lower priority.";
    }
    
    insightBox.innerHTML = `
        <div class="insight-title">${title}</div>
        <p>${body}</p>
    `;
}

// Handle Form Submission for Wellbeing Model
async function handleWellbeingSubmit(event) {
    event.preventDefault();
    
    const select = document.getElementById("wellbeing-entity");
    if (!select.value) {
        alert("Please select a Country / Region from the list.");
        return;
    }
    
    // Show Loading state
    const submitBtn = document.getElementById("wellbeing-submit");
    submitBtn.innerText = "Executing Projections...";
    submitBtn.disabled = true;
    
    const requestData = {
        Entity: select.value,
        Year: parseInt(document.getElementById("wellbeing-year").value)
    };
    
    try {
        const res = await fetch(`${API_BASE}/api/predict/wellbeing`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData)
        });
        
        if (!res.ok) {
            const errDetails = await res.json();
            throw new Error(errDetails.detail || "Server error");
        }
        
        const result = await res.json();
        
        // Display Results
        document.getElementById("wellbeing-placeholder").classList.add("hidden");
        document.getElementById("wellbeing-result").classList.remove("hidden");
        
        const val = result.prediction;
        document.getElementById("wellbeing-pred-val").innerText = `${val}%`;
        document.getElementById("wellbeing-explanation").innerText = result.explanation;
        
        // Update bar indicator (most values between 1% and 15% DALYs, scaled accordingly)
        const indicatorWidth = Math.min(100, (val / 15) * 100);
        document.getElementById("wellbeing-indicator-fill").style.width = `${indicatorWidth}%`;
        
        // Generate Wellbeing Insights
        generateWellbeingInsights(val, requestData);
        
    } catch (err) {
        alert("Wellbeing Prediction Failed: " + err.message);
    } finally {
        submitBtn.innerText = "Predict Mental DALYs Share";
        submitBtn.disabled = false;
    }
}

// Generate Wellbeing Insights Panel HTML
function generateWellbeingInsights(val, request) {
    const insightBox = document.getElementById("wellbeing-insight");
    let title = "";
    let body = "";
    
    if (val > 10.0) {
        title = "🚨 High Mental Disorder Burden";
        body = `In ${request.Year}, mental health conditions represent an elevated portion (<b>${val.toFixed(2)}%</b>) of the total Disability-Adjusted Life Years (DALYs) in ${request.Entity}. This suggests a critical need for targeted psychiatric resources, counseling facilities, and systemic mental wellness frameworks.`;
    } else if (val > 5.0) {
        title = "⚠️ Moderate Mental Disorder Burden";
        body = `Mental health disorder DALYs account for <b>${val.toFixed(2)}%</b> of the burden in ${request.Entity}. Focus should remain on strengthening community support networks and accessible cognitive counseling channels.`;
    } else {
        title = "🟢 Low Reported Burden Rate";
        body = `Mental health disorder DALYs account for <b>${val.toFixed(2)}%</b> of total DALYs in ${request.Entity}. While lower than global averages, continued monitoring and reduction of reporting barriers or social stigmas is recommended.`;
    }
    
    insightBox.innerHTML = `
        <div class="insight-title">${title}</div>
        <p>${body}</p>
    `;
}
