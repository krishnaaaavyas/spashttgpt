// Intersection Observer for scroll animations
const observerOptions = {
  threshold: 0.1,
  rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('animate');
      // For stagger animations
      if (entry.target.classList.contains('stagger-container')) {
        const items = entry.target.querySelectorAll('.stagger-item');
        items.forEach((item, index) => {
          setTimeout(() => {
            item.classList.add('animate');
          }, index * 100);
        });
      }
    }
  });
}, observerOptions);

// Initialize scroll animations
document.addEventListener('DOMContentLoaded', () => {
  // Observe all elements with scroll animation classes
  const animatedElements = document.querySelectorAll(
    '.scroll-animate, .fade-in-up, .fade-in-left, .fade-in-right, .scale-in, .rotate-in, .slide-in-bottom, .stagger-container'
  );
  
  animatedElements.forEach(el => observer.observe(el));
  
  // Scroll progress indicator
  const progressBar = document.querySelector('.scroll-progress');
  if (progressBar) {
    window.addEventListener('scroll', () => {
      const scrollPercent = (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100;
      progressBar.style.width = `${scrollPercent}%`;
    });
  }
  
  // Scroll to top button
  const scrollToTopBtn = document.querySelector('.scroll-to-top');
  if (scrollToTopBtn) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 300) {
        scrollToTopBtn.classList.add('visible');
      } else {
        scrollToTopBtn.classList.remove('visible');
      }
    });
    
    scrollToTopBtn.addEventListener('click', () => {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  }
  
  // Parallax effect
  const parallaxElements = document.querySelectorAll('.parallax-element');
  if (parallaxElements.length > 0) {
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      parallaxElements.forEach(element => {
        const rate = scrolled * -0.5;
        element.style.transform = `translateY(${rate}px)`;
      });
    });
  }
});

// Enhanced message animation for chat
export const animateMessage = (messageElement, isUser = false) => {
  messageElement.classList.add('message-enter');
  if (isUser) {
    messageElement.classList.add('user');
  }
  
  // Remove animation class after animation completes
  setTimeout(() => {
    messageElement.classList.remove('message-enter', 'user');
  }, 500);
};

// Smooth scroll to element
export const smoothScrollTo = (element, offset = 0) => {
  const elementPosition = element.offsetTop - offset;
  window.scrollTo({
    top: elementPosition,
    behavior: 'smooth'
  });
};
