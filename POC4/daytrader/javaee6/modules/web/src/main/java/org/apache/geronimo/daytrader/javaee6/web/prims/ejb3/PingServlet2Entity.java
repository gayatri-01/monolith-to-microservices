/**
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package org.apache.geronimo.daytrader.javaee6.web.prims.ejb3;

import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import javax.servlet.annotation.WebServlet;

import javax.persistence.EntityManager;
import javax.persistence.PersistenceContext;
import javax.ejb.EJB;


import org.apache.geronimo.daytrader.javaee6.entities.QuoteDataBean;
import org.apache.geronimo.daytrader.javaee6.utils.Log;
import org.apache.geronimo.daytrader.javaee6.utils.TradeConfig;

/**
 * 
 * Primitive designed to run within the TradeApplication and makes use of
 * {@link trade_client.TradeConfig} for config parameters and random stock
 * symbols. Servlet will generate a random stock symbol and get the price of
 * that symbol using a {@link trade.Quote} Entity EJB This tests the common path
 * of a Servlet calling an Entity EJB to get data
 * 
 */
@WebServlet("/ejb3/PingServlet2Entity")
public class PingServlet2Entity extends HttpServlet {
    private static String initTime;

    private static int hitCount;
    
//    private @EJB QuoteDataBean quote;
    private QuoteDataBean quote;

    @PersistenceContext(unitName = "daytrader")
    private EntityManager em;

    public void doPost(HttpServletRequest req, HttpServletResponse res) throws ServletException, IOException {
        doGet(req, res);
    }

    public void doGet(HttpServletRequest req, HttpServletResponse res) throws IOException, ServletException {

        res.setContentType("text/html");
        java.io.PrintWriter out = res.getWriter();

        String symbol = null;

        StringBuffer output = new StringBuffer(100);
        output.append("<html><head><title>Servlet2Entity</title></head>" + "<body><HR><FONT size=\"+2\" color=\"#000066\">PingServlet2Entity<BR></FONT>" + "<FONT size=\"-1\" color=\"#000066\"><BR>PingServlet2Entity accesses an EntityManager"
                + " using a PersistenceContext annotaion and then gets the price of a random symbol (generated by TradeConfig)" + " through the EntityManager find method");
        try {
            // generate random symbol
            try {
                int iter = TradeConfig.getPrimIterations();
                for (int ii = 0; ii < iter; ii++) {
                    // get a random symbol to look up and get the key to that
                    // symbol.
                    symbol = TradeConfig.rndSymbol();
                    // find the EntityInstance.
                    quote = (QuoteDataBean) em.find(QuoteDataBean.class, symbol);
                }
            } catch (Exception e) {
                Log.error("web_primtv.PingServlet2Entity.doGet(...): error performing find");
                throw e;
            }
            // get the price and print the output.
            output.append("<HR>initTime: " + initTime + "<BR>Hit Count: ").append(hitCount++);
            output.append("<HR>Quote Information<BR><BR> " + quote.toHTML());
            output.append("</font><HR></body></html>");
            out.println(output.toString());
        } catch (Exception e) {
            Log.error(e, "PingServlet2Entity.doGet(...): error");
            // this will send an Error to teh web applications defined error
            // page.
            res.sendError(500, "PingServlet2Entity.doGet(...): error" + e.toString());

        }
    }

    public String getServletInfo() {
        return "web primitive, tests Servlet to Entity EJB path";
    }

    public void init(ServletConfig config) throws ServletException {
        super.init(config);
        hitCount = 0;
        initTime = new java.util.Date().toString();
    }
}