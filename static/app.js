/**
 * VNTech Legal AI — Chatbot Frontend Logic
 * Supports SSE Token Streaming, Markdown Rendering, Image Drag & Drop Upload
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatMessages = document.getElementById('chatMessages');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const btnSend = document.getElementById('btnSend');
    const btnAttach = document.getElementById('btnAttach');
    const fileInput = document.getElementById('fileInput');
    const filePreviewBar = document.getElementById('filePreviewBar');
    const previewFileName = document.getElementById('previewFileName');
    const btnRemoveFile = document.getElementById('btnRemoveFile');
    const btnClearChat = document.getElementById('btnClearChat');
    const btnExportChat = document.getElementById('btnExportChat');
    const dropzoneOverlay = document.getElementById('dropzoneOverlay');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');
    const modeBtns = document.querySelectorAll('.mode-btn');

    let selectedFile = null;
    let isGenerating = false;
    let chatHistory = [];
    let currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
    });

    // Enter to submit (Shift+Enter for new line)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Suggestion chips
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.getAttribute('data-prompt');
            if (prompt && !isGenerating) {
                userInput.value = prompt;
                chatForm.dispatchEvent(new Event('submit'));
            }
        });
    });

    // Mode buttons
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Clear Chat
    btnClearChat.addEventListener('click', () => {
        if (confirm('Bạn có chắc chắn muốn làm mới toàn bộ cuộc trò chuyện không?')) {
            chatHistory = [];
            currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
            chatMessages.innerHTML = `
                <div class="welcome-card">
                    <div class="welcome-icon"><i class="fa-solid fa-scale-balanced"></i></div>
                    <h1>Xin chào! Tôi có thể hỗ trợ gì cho bạn hôm nay?</h1>
                    <p>Hệ thống AI tự động tra cứu từ <strong>11 bộ luật Việt Nam</strong>, cơ sở dữ liệu nhân sự <strong>VNTech</strong> và rà soát bẫy pháp lý trong <strong>hợp đồng / ảnh chụp</strong>.</p>
                </div>
            `;
        }
    });

    // Export Chat
    btnExportChat.addEventListener('click', () => {
        if (chatHistory.length === 0) {
            alert('Chưa có nội dung trò chuyện để xuất.');
            return;
        }
        let transcript = `# VNTech Legal AI — Lịch sử Tư vấn Pháp lý\n\n`;
        chatHistory.forEach(item => {
            transcript += `### ${item.role === 'user' ? '👤 Người dùng' : '🤖 Trợ lý AI (' + (item.intent || 'RAG') + ')'}:\n${item.text}\n\n---\n\n`;
        });
        const blob = new Blob([transcript], { type: 'text/markdown;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `legal_chat_export_${new Date().toISOString().slice(0, 10)}.md`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // File Attach Button
    btnAttach.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    btnRemoveFile.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        filePreviewBar.style.display = 'none';
    });

    function handleSelectedFile(file) {
        selectedFile = file;
        previewFileName.textContent = file.name;
        filePreviewBar.style.display = 'flex';
        userInput.focus();
    }

    // Drag and Drop
    window.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzoneOverlay.classList.add('active');
    });

    dropzoneOverlay.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzoneOverlay.classList.remove('active');
    });

    dropzoneOverlay.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzoneOverlay.classList.remove('active');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    // =========================================================================
    // Message Handling & Streaming
    // =========================================================================

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const messageText = userInput.value.trim();

        if ((!messageText && !selectedFile) || isGenerating) {
            return;
        }

        // Remove welcome card on first message
        const welcomeCard = document.querySelector('.welcome-card');
        if (welcomeCard) {
            welcomeCard.remove();
        }

        isGenerating = true;
        btnSend.disabled = true;

        // 1. Render User Message
        let displayUserMsg = messageText;
        if (selectedFile) {
            displayUserMsg = `[File đính kèm: 📁 ${selectedFile.name}]\n` + (messageText || "Yêu cầu thẩm định rủi ro hợp đồng trong file này.");
        }
        appendMessage('user', displayUserMsg);
        chatHistory.push({ role: 'user', text: displayUserMsg });

        // Reset Input
        userInput.value = '';
        userInput.style.height = 'auto';

        // 2. Prepare AI Message Placeholder
        const aiMessageObj = appendMessage('ai', '', true);
        const fileToUpload = selectedFile;
        selectedFile = null;
        filePreviewBar.style.display = 'none';
        fileInput.value = '';

        try {
            if (fileToUpload) {
                // Upload Endpoint with streaming
                await handleFileUploadStream(fileToUpload, messageText, aiMessageObj);
            } else {
                // Chat Endpoint with streaming
                await handleChatStream(messageText, aiMessageObj);
            }
        } catch (err) {
            aiMessageObj.bubble.innerHTML = `<span style="color: #f43f5e;"><i class="fa-solid fa-triangle-exclamation"></i> Lỗi kết nối: ${err.message}</span>`;
        } finally {
            aiMessageObj.bubble.classList.remove('typing-cursor');
            isGenerating = false;
            btnSend.disabled = false;
            userInput.focus();
        }
    });

    async function handleChatStream(message, aiMsgObj) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        await processSSEStream(response.body, aiMsgObj);
    }

    async function handleFileUploadStream(file, notes, aiMsgObj) {
        const formData = new FormData();
        formData.append('file', file);
        if (notes) formData.append('notes', notes);

        const response = await fetch('/api/upload-contract', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        await processSSEStream(response.body, aiMsgObj);
    }

    async function processSSEStream(streamBody, aiMsgObj) {
        const reader = streamBody.getReader();
        const decoder = new TextDecoder('utf-8');
        let fullText = '';
        let currentIntent = 'LEGAL_RAG';
        let renderPending = false;

        function scheduleRender() {
            if (!renderPending) {
                renderPending = true;
                requestAnimationFrame(() => {
                    aiMsgObj.bubble.innerHTML = marked.parse(fullText);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    renderPending = false;
                });
            }
        }

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr) continue;

                    try {
                        const data = JSON.parse(jsonStr);

                        if (data.type === 'intent') {
                            currentIntent = data.intent;
                            updateIntentBadge(aiMsgObj.badge, currentIntent);
                        } else if (data.type === 'token') {
                            fullText += data.token;
                            // Throttled 60FPS Markdown rendering
                            scheduleRender();
                        } else if (data.type === 'done') {
                            break;
                        }
                    } catch (e) {
                        // ignore chunk json parsing errors
                    }
                }
            }
        }

        // Final render to ensure complete and pristine Markdown
        aiMsgObj.bubble.innerHTML = marked.parse(fullText);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        chatHistory.push({ role: 'ai', intent: currentIntent, text: fullText });
    }

    function appendMessage(role, content = '', isStreaming = false) {
        const row = document.createElement('div');
        row.className = `message-row ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = role === 'user' ? '<i class="fa-regular fa-user"></i>' : '<i class="fa-solid fa-scale-balanced"></i>';

        const wrapper = document.createElement('div');
        wrapper.className = 'message-content-wrapper';

        let badge = null;
        if (role === 'ai') {
            badge = document.createElement('span');
            badge.className = 'intent-badge LEGAL_RAG';
            badge.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Đang xử lý...';
            wrapper.appendChild(badge);
        }

        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${isStreaming ? 'typing-cursor' : ''}`;
        bubble.innerHTML = marked.parse(content || '');

        wrapper.appendChild(bubble);

        // Actions bar
        if (role === 'ai') {
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            actions.innerHTML = `
                <button class="btn-msg-action" title="Sao chép câu trả lời">
                    <i class="fa-regular fa-copy"></i> Sao chép
                </button>
            `;
            const btnCopy = actions.querySelector('.btn-msg-action');
            btnCopy.addEventListener('click', () => {
                navigator.clipboard.writeText(bubble.innerText).then(() => {
                    btnCopy.innerHTML = '<i class="fa-solid fa-check"></i> Đã chép';
                    setTimeout(() => {
                        btnCopy.innerHTML = '<i class="fa-regular fa-copy"></i> Sao chép';
                    }, 2000);
                });
            });
            wrapper.appendChild(actions);
        }

        row.appendChild(avatar);
        row.appendChild(wrapper);
        chatMessages.appendChild(row);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return { row, bubble, badge };
    }

    function updateIntentBadge(badgeElement, intent) {
        if (!badgeElement) return;

        badgeElement.className = `intent-badge ${intent}`;
        let icon = 'fa-book-open';
        let label = 'Căn cứ Pháp luật (LEGAL CHATBOT)';

        if (intent === 'GREETING') {
            icon = 'fa-comments';
            label = 'Chào mừng & Giới thiệu';
        } else if (intent === 'HR_DATABASE') {
            icon = 'fa-users';
            label = 'Nhân sự VNTech (HR Database)';
        } else if (intent === 'CONTRACT_RISK') {
            icon = 'fa-file-shield';
            label = 'Thẩm định Rủi ro Hợp đồng';
        } else if (intent === 'HYBRID') {
            icon = 'fa-network-wired';
            label = 'Kết hợp (Nhân sự + Pháp luật)';
        }


        badgeElement.innerHTML = `<i class="fa-solid ${icon}"></i> ${label}`;
    }
});
