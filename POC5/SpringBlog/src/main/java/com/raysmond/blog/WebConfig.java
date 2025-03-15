package com.raysmond.blog;

import com.raysmond.blog.support.web.ViewHelper;

import jakarta.annotation.PostConstruct;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.env.Environment;
import org.springframework.security.web.csrf.CsrfToken;
import org.springframework.web.servlet.HandlerInterceptor;
import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import javax.servlet.http.HttpServletResponse;

import static com.raysmond.blog.Constants.ENV_DEVELOPMENT;
import static com.raysmond.blog.Constants.ENV_PRODUCTION;

import lombok.extern.slf4j.Slf4j;

/**
 * \@author Raysmond .
 */
@Configuration
@Slf4j
public class WebConfig implements WebMvcConfigurer {
    @Autowired
    private ViewHelper viewHelper;

    @Autowired
    private Environment env;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(viewObjectAddingInterceptor());
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        if (env.acceptsProfiles(ENV_DEVELOPMENT)) {
            log.debug("Register CORS configuration");
            registry.addMapping("/api/**")
                    .allowedOrigins("http://localhost:8080")
                    .allowedMethods("*")
                    .allowedHeaders("*")
                    .allowCredentials(true)
                    .maxAge(3600);
        }
    }

    @PostConstruct
    public void registerJadeViewHelpers() {
        viewHelper.setApplicationEnv(this.getApplicationEnv());
    }

    @Bean
    public HandlerInterceptor viewObjectAddingInterceptor() {
        return new HandlerInterceptor() {

            public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler)
                    throws Exception {
                viewHelper.setStartTime(System.currentTimeMillis());
                return true;
            }


            public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler,
                                   ModelAndView view) {
                CsrfToken token = (CsrfToken) request.getAttribute(CsrfToken.class.getName());
                if (token != null && view != null) {
                    view.addObject(token.getParameterName(), token);
                }
            }
        };
    }

    public String getApplicationEnv() {
        return this.env.acceptsProfiles(ENV_PRODUCTION) ? ENV_PRODUCTION : ENV_DEVELOPMENT;
    }
}