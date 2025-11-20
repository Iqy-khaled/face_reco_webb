// Face Recognition Attendance System - Main JavaScript
document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const resultDiv = document.getElementById('result');
    const resultPhoto = document.getElementById('result-photo');
    const recognitionInfo = document.getElementById('recognition-info');
    const snapResult = document.getElementById('snap-result');

    // Initialize camera
    initCamera();

    // Event listeners for tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            activateTab(this.getAttribute('data-tab'));
        });
    });

    // Initialize the first tab
    activateTab('recognition');

    // Functions
    function initCamera() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => {
                showResult(`Error accessing webcam: ${err}`, 'alert-danger');
            });
    }

    function activateTab(tabId) {
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.getElementById(tabId).classList.add('active');
        document.querySelector(`.tab[data-tab="${tabId}"]`).classList.add('active');
    }

    function showResult(message, className) {
        resultDiv.innerText = message;
        resultDiv.className = 'alert';
        resultDiv.classList.add(className);
        resultDiv.style.display = 'block';
    }

    window.captureImage = function(isCheckOut) {
        resultPhoto.style.display = 'none';
        recognitionInfo.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        showResult('Processing...', 'alert-info');
        resultPhoto.src = imageData;
        resultPhoto.style.display = 'block';
        snapResult.style.display = 'block';

        fetch('/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData, is_check_out: isCheckOut })
        })
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showResult(`Error: ${data.error}`, 'alert-danger');
            } else {
                const actionText = data.is_check_out ? 'Checked Out' : 'Checked In';
                showResult(`${actionText}: ${data.name} (Confidence: ${(data.confidence * 100).toFixed(2)}%) at ${new Date(data.timestamp).toLocaleTimeString()}`, 'alert-success');
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

    window.registerPerson = function() {
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
        showResult('Registering...', 'alert-info');
        resultPhoto.src = imageData;
        resultPhoto.style.display = 'block';
        snapResult.style.display = 'block';

        fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, image: imageData })
        })
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            return response.json();
        })
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
            }
        })
        .catch(err => {
            showResult(`Error: ${err}`, 'alert-danger');
        });
    };

    window.loadAttendanceData = function() {
        const attendanceTable = document.getElementById('attendance-table');
        attendanceTable.innerHTML = '<tr><td colspan="5">Loading...</td></tr>';
        fetch('/api/attendance/today')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
                return response.json();
            })
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
                    let duration = checkOut ? `${Math.floor((checkOut - checkIn) / 1000 / 60 / 60)}h ${Math.floor(((checkOut - checkIn) / 1000 / 60) % 60)}m` : 'Still present';
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

    window.showFaces = function() {
        resultDiv.innerText = 'Loading faces...';
        resultDiv.classList.remove('alert-danger', 'alert-success');
        resultDiv.classList.add('alert-info');
        document.getElementById('faces').innerHTML = '';
        fetch('/api/people')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
                return response.json();
            })
            .then(people => {
                if (people.length === 0) {
                    showResult('No registered people found', 'alert-warning');
                    return;
                }
                people.forEach(person => {
                    fetch(`/api/face-image/${person.person_id}`)
                        .then(response => {
                            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
                            return response.json();
                        })
                        .then(data => {
                            if (data.error) return;
                            const div = document.createElement('div');
                            div.className = 'border p-4 rounded bg-gray-50';
                            div.innerHTML = `
                                <img src="${data.image}" alt="${person.name}" class="w-full h-32 object-cover rounded mb-2">
                                <p><strong>Name:</strong> ${person.name}</p>
                                <p><strong>Email:</strong> ${person.email || 'N/A'}</p>
                                <p><strong>Department:</strong> ${person.department || 'N/A'}</p>
                            `;
                            document.getElementById('faces').appendChild(div);
                        })
                        .catch(err => console.error(`Error fetching image for ${person.name}: ${err}`));
                });
                showResult('Faces loaded successfully', 'alert-success');
            })
            .catch(err => showResult(`Error loading faces: ${err}`, 'alert-danger'));
    };

    window.showImageSizes = function() {
        resultDiv.innerText = 'Loading image size statistics...';
        resultDiv.classList.remove('alert-danger', 'alert-success');
        resultDiv.classList.add('alert-info');
        document.getElementById('faces').innerHTML = '';
        fetch('/api/image-sizes')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
                return response.json();
            })
            .then(data => {
                if (data.images.length === 0) {
                    showResult('No image data found', 'alert-warning');
                    return;
                }
                const summaryDiv = document.createElement('div');
                summaryDiv.className = 'border p-4 rounded bg-blue-50 mb-4';
                summaryDiv.innerHTML = `
                    <h3 class="text-xl font-bold mb-2">Image Storage Summary</h3>
                    <div class="grid grid-cols-3 gap-4">
                        <div>
                            <p><strong>Total Images:</strong> ${data.summary.total_images}</p>
                            <p><strong>Total Size:</strong> ${data.summary.total_size_mb.toFixed(2)} MB</p>
                        </div>
                        <div>
                            <p><strong>Average Size:</strong> ${data.summary.avg_size_kb.toFixed(2)} KB</p>
                        </div>
                    </div>
                `;
                document.getElementById('faces').appendChild(summaryDiv);
                const tableDiv = document.createElement('div');
                tableDiv.className = 'overflow-x-auto';
                tableDiv.innerHTML = `
                    <table class="min-w-full bg-white border">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="py-2 px-4 border-b text-left">Person ID</th>
                                <th class="py-2 px-4 border-b text-left">Name</th>
                                <th class="py-2 px-4 border-b text-right">Size (KB)</th>
                                <th class="py-2 px-4 border-b text-right">Size (Bytes)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.images.map(img => `
                                <tr>
                                    <td class="py-2 px-4 border-b">${img.person_id}</td>
                                    <td class="py-2 px-4 border-b">${img.name}</td>
                                    <td class="py-2 px-4 border-b text-right">${img.image_size_kb.toFixed(2)}</td>
                                    <td class="py-2 px-4 border-b text-right">${img.image_size}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `;
                document.getElementById('faces').appendChild(tableDiv);
                showResult('Image size statistics loaded successfully', 'alert-success');
            })
            .catch(err => showResult(`Error loading image sizes: ${err}`, 'alert-danger'));
    };

    window.importKnownFaces = function() {
        if (!confirm('This will import face images from the known_faces directory. Continue?')) return;
        resultDiv.innerText = 'Importing known faces...';
        resultDiv.classList.remove('alert-danger', 'alert-success');
        resultDiv.classList.add('alert-info');
        document.getElementById('faces').innerHTML = '';
        fetch('/api/import-known-faces', { method: 'POST' })
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
                return response.json();
            })
            .then(data => {
                const summaryDiv = document.createElement('div');
                summaryDiv.className = 'border p-4 rounded bg-blue-50 mb-4';
                summaryDiv.innerHTML = `
                    <h3 class="text-xl font-bold mb-2">Import Results</h3>
                    <p class="mb-2">${data.message}</p>
                    <div class="grid grid-cols-3 gap-4">
                        <div class="bg-green-100 p-2 rounded"><p class="font-bold text-green-800">Imported: ${data.imported}</p></div>
                        <div class="bg-yellow-100 p-2 rounded"><p class="font-bold text-yellow-800">Skipped: ${data.skipped}</p></div>
                        <div class="bg-red-100 p-2 rounded"><p class="font-bold text-red-800">Errors: ${data.error}</p></div>
                    </div>
                `;
                document.getElementById('faces').appendChild(summaryDiv);
                if (data.details && data.details.length > 0) {
                    const tableDiv = document.createElement('div');
                    tableDiv.className = 'overflow-x-auto';
                    tableDiv.innerHTML = `
                        <h3 class="text-lg font-bold mb-2">Import Details</h3>
                        <table class="min-w-full bg-white border">
                            <thead class="bg-gray-100">
                                <tr>
                                    <th class="py-2 px-4 border-b text-left">File</th>
                                    <th class="py-2 px-4 border-b text-left">Status</th>
                                    <th class="py-2 px-4 border-b text-left">Details</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.details.map(item => {
                                    let statusClass = item.status === 'imported' ? 'text-green-600' : item.status === 'skipped' ? 'text-yellow-600' : 'text-red-600';
                                    let details = item.status === 'imported' ? `Person ID: ${item.person_id}, Name: ${item.name}` : item.reason || '';
                                    return `
                                        <tr>
                                            <td class="py-2 px-4 border-b">${item.file}</td>
                                            <td class="py-2 px-4 border-b ${statusClass} font-semibold">${item.status}</td>
                                            <td class="py-2 px-4 border-b">${details}</td>
                                        </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    `;
                    document.getElementById('faces').appendChild(tableDiv);
                }
                const instructionsDiv = document.createElement('div');
                instructionsDiv.className = 'mt-4 p-4 bg-gray-50 rounded';
                instructionsDiv.innerHTML = `
                    <h3 class="font-bold">How to add more faces:</h3>
                    <ol class="list-decimal pl-5 mt-2">
                        <li>Place image files in the <code>known_faces</code> directory</li>
                        <li>Name each file with the person's name (e.g., <code>John_Smith.jpg</code>)</li>
                        <li>Each image should contain exactly one face</li>
                        <li>Click "Import Known Faces" button to import them</li>
                    </ol>
                `;
                document.getElementById('faces').appendChild(instructionsDiv);
                showResult(data.imported > 0 ? `Success! Imported ${data.imported} faces` : data.message, data.imported > 0 ? 'alert-success' : data.error > 0 ? 'alert-danger' : 'alert-warning');
            })
            .catch(err => showResult(`Error importing known faces: ${err}`, 'alert-danger'));
    };
});
