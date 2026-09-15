---
layout: page
title: Projects
subtitle: Research and engineering projects
permalink: /projects/
---

<div class="posts-list">
{% assign project_posts = site.posts | where: "category", "project" %}
{% if project_posts.size > 0 %}
  {% for project in project_posts %}
  <article class="post-preview">
    <div class="post-entry-container">
      {% if project.thumbnail-img %}
      <div class="post-image">
        <a href="{{ project.url | relative_url }}">
          <img src="{{ project.thumbnail-img | relative_url }}" alt="{{ project.title }}">
        </a>
      </div>
      {% endif %}
      <div class="post-entry">
        <h2 class="post-title"><a href="{{ project.url | relative_url }}">{{ project.title }}</a></h2>

        {% if project.subtitle %}
        <p class="post-subtitle">{{ project.subtitle }}</p>
        {% endif %}

        <p class="post-meta">{{ project.date | date: "%B %Y" }}</p>

        {% if project.tags %}
        <div class="post-tags">
          {% for tag in project.tags limit: 5 %}
          <span class="post-tag">{{ tag }}</span>
          {% endfor %}
        </div>
        {% endif %}
      </div>
    </div>
  </article>
  {% endfor %}
{% else %}
  <p><em>No projects yet.</em></p>
{% endif %}
</div>
