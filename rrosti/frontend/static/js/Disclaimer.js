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

export class Disclaimer {
    constructor(onAccept, accepted) {
        this.disclaimercontent = 'Dummy disclaimer.';

        this.accepted = accepted;
        this.onAccept = onAccept;
        this.element = this.renderElement();
    }

    onAcceptClick(event) {
        this.button.disabled = true;
        this.button.classList.add('accepted');
        this.button.innerText = "akzeptiert";

        this.disclaimerDetails.removeAttribute('open');

        this.onAccept();
    }

    renderRenderAcceptButton() {
        var buttonWrapper = document.createElement('div');
        buttonWrapper.classList.add('accept-button-wrapper');

        this.button = document.createElement('div');
        this.button.classList.add('accept-button');
        this.button.innerText = "Ich akzeptiere";

        buttonWrapper.append(this.button);
        this.button.addEventListener('click', this.onAcceptClick.bind(this));

        return buttonWrapper;
    }

    renderElement() {
        var rootElement = document.createElement('div');
        rootElement.classList.add("message");
        rootElement.classList.add("answer");
        rootElement.classList.add("disclaimer");

        var bubbleElement = document.createElement('div');
        bubbleElement.classList.add('bubble');

        this.disclaimerDetails = document.createElement('details');
        this.disclaimerDetails.open = true;
        this.disclaimerDetails.innerHTML = `<summary>Disclaimer</summary><p class='answer-content'>${this.disclaimercontent}</p></details>`;

        rootElement.append(bubbleElement);
        bubbleElement.append(this.disclaimerDetails);
        bubbleElement.append(this.renderRenderAcceptButton());

        if(this.accepted) {
            this.onAcceptClick();
        }

        return rootElement;
    }
}
