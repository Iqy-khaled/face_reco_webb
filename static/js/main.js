// Face Recognition Attendance System - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Base URL for all API calls
    const BASE_URL = 'http://localhost:5000';
    
    const video = document.getElementById('video');
    const resultDiv = document.getElementById('result');
    const resultPhoto = document.getElementById('result-photo');
    const recognitionInfo = document.getElementById('recognition-info');
    const snapResult = document.getElementById('snap-result');
    const facesDiv = document.getElementById('faces-container'); // Added for showFaces function

    // Initialize camera
    initCamera();

    // Event listeners for tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            activateTab(this.getAttribute('data-tab'));
        });
    });

    // Initialize the first tab
    if (document.querySelector('.tab')) {
        activateTab(document.querySelector('.tab').getAttribute('data-tab'));
    }

    // Functions

    // Function to initialize the camera
    function initCamera() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                showResult(`Error accessing webcam: ${err}`, 'alert-danger');
            });
    }

    // Function to activate a tab
    function activateTab(tabId) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Deactivate all tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Activate selected tab and content
        document.getElementById(tabId).classList.add('active');
        document.querySelector(`.tab[data-tab="${tabId}"]`).classList.add('active');

        // Load faces if on the "people" tab
        if (tabId === 'people-tab') {
            showFaces();
        }
    }

    // Function to show result message
    function showResult(message, className) {
        resultDiv.innerText = message;
        resultDiv.className = 'alert';
        resultDiv.classList.add(className);
        resultDiv.style.display = 'block';
    }

    // Function to capture image
    window.captureImage = function(isCheckOut) {
        // Clear previous results
        resultPhoto.style.display = 'none';
        recognitionInfo.innerHTML = '';
        
        // Create canvas for snapshot
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Show processing message
        showResult('Processing...', 'alert-info');

        // Display the snapshot
        resultPhoto.src = imageData;
        resultPhoto.style.display = 'block';
        snapResult.style.display = 'block';

        fetch(`${BASE_URL}/api/recognize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, is_check_out: isCheckOut })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showResult(`Error: ${data.error}`, 'alert-danger');
            } else {
                const actionText = data.is_check_out ? 'Checked Out' : 'Checked In';
                showResult(`${actionText}: ${data.name} (Confidence: ${(data.confidence * 100).toFixed(2)}%) at ${new Date(data.timestamp).toLocaleTimeString()}`, 'alert-success');
                
                // Show recognition badge
                const badgeClass = data.confidence > 0.7 ? 'badge-success' : 'badge-warning';
                recognitionInfo.innerHTML = `
                    <div class="recognition-badge ${badgeClass}">
                        ${(data.confidence * 100).toFixed(2)}% Match
                    </div>
                    <h3>${data.name}</h3>
                    <p>ID: ${data.person_id} | ${data.pos || 'No Position'}</p>
                    <p>${actionText} at ${new Date(data.timestamp).toLocaleTimeString()}</p>
                `;
            }
        })
        .catch(err => {
            showResult(`Error: ${err}`, 'alert-danger');
        });
    };

    // Function to register a new person
    window.registerPerson = function() {
        // Clear previous results
        resultPhoto.style.display = 'none';
        recognitionInfo.innerHTML = '';
        
        const name = prompt('Enter person name:');
        if (!name) {
            showResult('Registration cancelled', 'alert-warning');
            return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');

        // Show processing message
        showResult('Registering...', 'alert-info');
        
        // Display the snapshot
        resultPhoto.src = imageData;
        resultPhoto.style.display = 'block';
        snapResult.style.display = 'block';

        fetch(`${BASE_URL}/api/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showResult(`Error: ${data.error}`, 'alert-danger');
            } else {
                showResult(`Registered: ${name} (ID: ${data.person_id})`, 'alert-success');
                recognitionInfo.innerHTML = `
                    <div class="recognition-badge badge-success">Registration Successful</div>
                    <h3>${name}</h3>
                    <p>Person ID: ${data.person_id}</p>
                `;

                // Update faces list if on people tab
                if (document.getElementById('people-tab').classList.contains('active')) {
                    showFaces();
                }
            }
        })
        .catch(err => {
            showResult(`Error: ${err}`, 'alert-danger');
        });
    };

    // Function to show registered faces
    window.showFaces = function() {
        if (!facesDiv) return;

        showResult('Loading faces...', 'alert-info');
        facesDiv.innerHTML = '';

        fetch(`${BASE_URL}/api/people`)
            .then(response => response.json())
            .then(people => {
                if (people.length === 0) {
                    showResult('No registered people found', 'alert-warning');
                    return;
                }

                showResult(`Found ${people.length} registered people`, 'alert-success');

                people.forEach(person => {
                    fetch(`${BASE_URL}/api/face-image/${person.person_id}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                console.warn(`No image for ${person.name}: ${data.error}`);
                                return;
                            }

                            const div = document.createElement('div');
                            div.className = 'person-card';
                            div.innerHTML = `
                                <img src="${data.image}" alt="${person.name}" class="face-image">
                                <div class="person-info">
                                    <h3>${person.name}</h3>
                                    <p>ID: ${person.person_id}</p>
                                    ${person.department ? `<p>Department: ${person.department}</p>` : ''}
                                    ${person.role ? `<p>Role: ${person.role}</p>` : ''}
                                </div>
                            `;
                            facesDiv.appendChild(div);
                        })
                        .catch(err => {
                            console.error(`Error loading image for ${person.name}:`, err);
                        });
                });
            })
            .catch(err => {
                showResult(`Error loading people: ${err}`, 'alert-danger');
            });
    };

    // Function to load attendance data
    window.loadAttendanceData = function() {
        const attendanceTable = document.getElementById('attendance-table');
        if (!attendanceTable) return;
        
        attendanceTable.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';
        
        fetch(`${BASE_URL}/api/attendance/today`)
            .then(response => response.json())
            .then(data => {
                if (data.length === 0) {
                    attendanceTable.innerHTML = '<tr><td colspan="5">No attendance records for today</td></tr>';
                    return;
                }
                
                let tableHtml = `
                    <tr>
                        <th>Name</th>
                        <th>Check In</th>
                        <th>Check Out</th>
                        <th>Duration</th>
                        <th>Confidence</th>
                    </tr>
                `;
                
                data.forEach(record => {
                    const checkIn = new Date(record.check_in_time);
                    let checkOut = record.check_out_time ? new Date(record.check_out_time) : null;
                    let duration = '';
                    
                    if (checkOut) {
                        const diff = (checkOut - checkIn) / 1000 / 60;
                        const hours = Math.floor(diff / 60);
                        const minutes = Math.floor(diff % 60);
                        duration = `${hours}h ${minutes}m`;
                    } else {
                        duration = 'Still present';
                    }
                    
                    tableHtml += `
                        <tr>
                            <td>${record.name}</td>
                            <td>${checkIn.toLocaleTimeString()}</td>
                            <td>${checkOut ? checkOut.toLocaleTimeString() : '-'}</td>
                            <td>${duration}</td>
                            <td>${(record.confidence_score * 100).toFixed(2)}%</td>
                        </tr>
                    `;
                });
                
                attendanceTable.innerHTML = tableHtml;
            })
            .catch(err => {
                attendanceTable.innerHTML = `<tr><td colspan="5">Error loading data: ${err}</td></tr>`;
            });
    };
});
