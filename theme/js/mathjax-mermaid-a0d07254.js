// MathJax 3 with $...$ delimiter support
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
  }
};

(function() {
  var s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
  s.async = true;
  document.head.appendChild(s);
})();

// Mermaid
(function() {
  var s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
  s.onload = function() {
    mermaid.initialize({ startOnLoad: true, theme: 'default' });
  };
  document.head.appendChild(s);
})();
