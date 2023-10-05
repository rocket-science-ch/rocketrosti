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

export class Answer {
    constructor(id, message, websocketClient) {
        this.message = message;
        this.id = id;
        this.websocketClient = websocketClient;
        this.type = message.type;

        this.element = this.renderElement(message.content);
    }

    likingInteracted() {
        return this.element.classList.contains('liked') || this.element.classList.contains('disliking') || this.element.classList.contains('disliked');
    }

    onLikeClick(event) {
        if(!this.likingInteracted()) {
            this.websocketClient.sendLike(this.id);
            this.element.classList.add('liked');
        }
    }

    onDislikeClick(event) {
        if(!this.likingInteracted()) {
            this.element.classList.add('disliking');
            this.dislikeInput.focus();
        }
    }

    onDislikeInputKeyDown(event) {
        if (event.keyCode == 13) {
            this.submitDislike();
            this.dislikeInput.blur()
        }
    }

    submitDislike() {
        if(this.dislikeInput.value) {
            this.websocketClient.sendDislike(this.id, this.dislikeInput.value);
            this.element.classList.remove('disliking');
            this.element.classList.add('disliked');
        }
    }

    replaceReferences(rawText, excerpts) {
        const referenceRegex = /\[\[([0-9]+)\]\]/gm;

        let output = rawText;
        let match;
        while((match = referenceRegex.exec(rawText)) != null) {
            if(match.index == referenceRegex.lastIndex) {
                referenceRegex.lastIndex++;
            }

            if(match.length == 2) {
                let id = parseInt(match[1]);
                let footnote = `<span class="source" tooltip="${excerpts[id].title}"><a href="${excerpts[id].link}" target="_blank">[+]</a></span>`;
                output = output.replace(match[0], footnote);
            }
        }

        return output;
    }

    renderLikesDislikes() {
        var likesElements = document.createElement('div');
        likesElements.classList.add('likes');

        var likeElement = document.createElement('div');
        likeElement.classList.add('like');
        likeElement.innerHTML = "<img src='/static/images/like.png' />";
        likeElement.addEventListener('click', this.onLikeClick.bind(this));

        var dislikeInputElementWrapper = document.createElement('div');
        dislikeInputElementWrapper.classList.add('dislike-input-wrapper');

        this.dislikeInput = document.createElement('input');
        this.dislikeInput.setAttribute('enterkeyhint', 'done');
        this.dislikeInput.type = "text";
        this.dislikeInput.placeholder = "Tell us what you dislike...";
        this.dislikeInput.addEventListener('keydown', this.onDislikeInputKeyDown.bind(this));

        var dislikeSendElement = document.createElement('div');
        dislikeSendElement.classList.add('dislike-submit');
        dislikeSendElement.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path d="M16,464,496,256,16,48V208l320,48L16,304Z"/></svg>';
        dislikeSendElement.addEventListener('click', this.submitDislike.bind(this));

        dislikeInputElementWrapper.append(this.dislikeInput);
        dislikeInputElementWrapper.append(dislikeSendElement);

        var dislikeElement = document.createElement('div');
        dislikeElement.classList.add('dislike');
        dislikeElement.innerHTML = "<img src='/static/images/dislike.png' />";
        dislikeElement.addEventListener('click', this.onDislikeClick.bind(this));

        likesElements.append(likeElement);
        likesElements.append(dislikeInputElementWrapper);
        likesElements.append(dislikeElement);

        return likesElements;
    }

    renderElement(answerMessage) {
        var rootElement = document.createElement('div');
        rootElement.classList.add("message");
        rootElement.classList.add("answer");

        var bubbleElement = document.createElement('div');
        bubbleElement.classList.add('bubble');

        if(this.type == 'answer') {
            rootElement.classList.add("final");
            var referencedText = this.replaceReferences(answerMessage.final_answer.content, answerMessage.rtfm_excerpts);
            bubbleElement.innerHTML = "<p class='answer-content'>" + referencedText + "</p>";
        } else if (this.type == 'intermediate') {
            rootElement.classList.add("intermediate");
            bubbleElement.innerHTML = "<details><summary>Intermediate</summary><p class='answer-content'>" + answerMessage + "</p></details";
        }


        rootElement.append(bubbleElement);
        bubbleElement.append(this.renderLikesDislikes());

        return rootElement;
    }
}
