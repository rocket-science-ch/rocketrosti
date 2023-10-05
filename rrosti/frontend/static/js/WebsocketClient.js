/* Copyright (c) 2023 Rocket Science AG, Switzerland

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. */

export class WebsocketClient {
    constructor(socketUrl, onAnswer, onIntermediate, onUserTurn, onOpen, onClose) {
        this.socketUrl = socketUrl;

        this.onAnswer = onAnswer;
        this.onIntermediate = onIntermediate;
        this.onUserTurn = onUserTurn;

        this.onOpen = onOpen;
        this.onClose = onClose;

        this.connected = false;

        this.createSocket();
    }

    onSocketMessage(event) {
        var message = JSON.parse(event.data)
        console.log("WebSocket message received:", message);

        if (message.type == 'user_turn') {
            this.onUserTurn();
        } else if (message.type == 'intermediate') {
            this.onIntermediate(message);
        } else if (message.type == 'answer') {
            this.onAnswer(message);
        }
    }

    onSocketClose(event) {
        console.log("WebSocket is closed now.");
        this.connected = false;

        this.onClose();
        setTimeout(this.createSocket.bind(this), 1000);
    }

    onSocketOpen(event) {
        console.log("WebSocket is open now.");
        this.connected = true;

        this.onOpen();
    }

    createSocket() {
        console.log("Creating WebSocket...");
        this.socket = new WebSocket(this.socketUrl);

        // Define a function to handle the open event
        this.socket.onopen = this.onSocketOpen.bind(this);

        this.socket.onerror = (evt) => {
            console.log("WebSocket error." + evt.message);
        };

        this.socket.onclose = this.onSocketClose.bind(this);

        // Define a function to handle the message event
        this.socket.onmessage = this.onSocketMessage.bind(this);
    }

    sendInput(message) {
        if (this.socket === null || this.socket.readyState !== 1) {  // 1 = OPEN
            console.log("Rescheduling to send until socket is open:", message);
            setTimeout(this.sendInput, 750, message);
            return;
        }
        console.log('Sending message: "' + message + '"');
        this.socket.send(message);
    }

    sendQuestion(question) {
        var message = JSON.stringify({'id': question.id, 'type': 'question', 'content': question.content});
        this.sendInput(message);
    }

    sendLike(refId) {
        var id = crypto.randomUUID();
        var message = JSON.stringify({'id': id, 'type': 'like', 'content': '', 'refId': refId});
        this.sendInput(message);
    }

    sendDislike(refId, dislikeMessage) {
        var id = crypto.randomUUID();
        var message = JSON.stringify({'id': id, 'type': 'dislike', 'content': dislikeMessage, 'refId': refId});
        this.sendInput(message);
    }

    sendNewContextRequest() {
        var id = crypto.randomUUID();
        var message = JSON.stringify({'id': id, 'type': 'new_question'});
        this.sendInput(message);
    }
}
